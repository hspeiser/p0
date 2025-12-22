use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver};
use crossterm::event::{Event, KeyCode};
use crossterm::terminal;
use nalgebra::{
    Matrix3, Matrix4, Quaternion, Rotation3, SymmetricEigen, UnitQuaternion, Vector3, Vector4,
};
use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use std::fs;
use std::io::Read;
use std::net::UdpSocket;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{thread, time};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    imu_port: String,

    #[arg(long, default_value_t = 115200)]
    imu_baud: u32,

    #[arg(long, default_value = "127.0.0.1:5005")]
    cam_listen: String,

    #[arg(long, default_value = "127.0.0.1:5006")]
    robot_listen: String,

    #[arg(long, default_value = "127.0.0.1:5007")]
    cmd_send: String,

    #[arg(long, default_value = "calibration.json")]
    calib_path: PathBuf,

    #[arg(long, default_value_t = 100.0)]
    control_hz: f64,

    #[arg(long, default_value_t = 0.05)]
    speed_scale: f64,

    #[arg(long)]
    calibrate: bool,

    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Clone)]
struct ImuSample {
    t_recv: Instant,
    q_e_s: UnitQuaternion<f64>,
}

#[derive(Debug, Clone)]
struct CamSample {
    t_recv: Instant,
    conf: f64,
    q_c_h: UnitQuaternion<f64>,
    pinch: f64,
    valid: bool,
    pinch_valid: bool,
}

#[derive(Debug, Clone)]
struct RobotState {
    t_recv: Instant,
    ee_pos: [f64; 3],
    q_b_t: UnitQuaternion<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Calibration {
    q_s_h: [f64; 4],
    q_c_e: [f64; 4],
    q_e_h_ref: [f64; 4],
    q_b_t_ref: [f64; 4],
    ee_pos: [f64; 3],
    pinch_open: f64,
    pinch_close: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        let q_s_h = nominal_q_s_h();
        Self {
            q_s_h: quat_to_wxyz(&q_s_h),
            q_c_e: [1.0, 0.0, 0.0, 0.0],
            q_e_h_ref: [1.0, 0.0, 0.0, 0.0],
            q_b_t_ref: [1.0, 0.0, 0.0, 0.0],
            ee_pos: [0.2, 0.0, 0.2],
            pinch_open: 0.12,
            pinch_close: 0.03,
        }
    }
}

#[derive(Debug, Deserialize)]
struct HandMessage {
    kind: Option<String>,
    t: Option<f64>,
    conf: Option<f64>,
    x: Option<[f64; 3]>,
    y: Option<[f64; 3]>,
    z: Option<[f64; 3]>,
    pinch: Option<f64>,
    valid: Option<bool>,
    frame_valid: Option<bool>,
    pinch_valid: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RobotStateMessage {
    kind: Option<String>,
    t: Option<f64>,
    ee_pos: Option<[f64; 3]>,
    ee_quat: Option<[f64; 4]>,
}

#[derive(Debug, Serialize)]
struct CommandMessage {
    kind: &'static str,
    t: f64,
    ee_pos: [f64; 3],
    ee_rotvec: [f64; 3],
    gripper: f64,
}

#[derive(Debug, Deserialize)]
struct ImuJson {
    q: [f64; 4],
}

struct OneEuroFilter {
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
    x_prev: Option<f64>,
    dx_prev: f64,
    t_prev: Option<Instant>,
}

impl OneEuroFilter {
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    fn filter(&mut self, x: f64, t: Instant) -> f64 {
        let dt = self
            .t_prev
            .map(|prev| (t - prev).as_secs_f64())
            .unwrap_or(1.0 / 60.0)
            .max(1e-6);

        let x_prev = self.x_prev.unwrap_or(x);
        let dx = (x - x_prev) / dt;
        let dx_hat = lowpass(dx, self.dx_prev, alpha(self.d_cutoff, dt));
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let x_hat = lowpass(x, x_prev, alpha(cutoff, dt));

        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        self.t_prev = Some(t);
        x_hat
    }
}

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff.max(1e-6));
    1.0 / (1.0 + tau / dt)
}

fn lowpass(x: f64, x_prev: f64, a: f64) -> f64 {
    a * x + (1.0 - a) * x_prev
}

fn main() -> Result<()> {
    let args = Args::parse();
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })?;

    let (imu_tx, imu_rx) = unbounded();
    let (ctrl_tx, ctrl_rx) = unbounded();
    spawn_imu_thread(&args.imu_port, args.imu_baud, imu_tx)?;
    spawn_input_thread(ctrl_tx)?;

    let cam_socket = UdpSocket::bind(&args.cam_listen)
        .with_context(|| format!("bind cam socket {}", args.cam_listen))?;
    cam_socket.set_nonblocking(true)?;

    let robot_socket = UdpSocket::bind(&args.robot_listen)
        .with_context(|| format!("bind robot socket {}", args.robot_listen))?;
    robot_socket.set_nonblocking(true)?;

    let cmd_socket = UdpSocket::bind("0.0.0.0:0")?;
    cmd_socket.connect(&args.cmd_send)?;

    let mut calib = load_calibration(&args.calib_path).unwrap_or_default();
    if args.calibrate || !args.calib_path.exists() {
        calib = run_calibration(&imu_rx, &cam_socket, &robot_socket, calib)?;
        save_calibration(&args.calib_path, &calib)?;
    }

    let mut q_s_h = unit_quat_from_wxyz(calib.q_s_h);
    let mut q_c_e = unit_quat_from_wxyz(calib.q_c_e);
    let mut q_e_h_ref = unit_quat_from_wxyz(calib.q_e_h_ref);
    let mut q_b_t_ref = unit_quat_from_wxyz(calib.q_b_t_ref);
    let mut ee_pos = calib.ee_pos;

    let mut last_imu: Option<ImuSample> = None;
    let mut last_cam: Option<CamSample> = None;
    let mut last_robot: Option<RobotState> = None;

    let mut q_imu_smooth: Option<UnitQuaternion<f64>> = None;
    let mut q_f_h_smooth: Option<UnitQuaternion<f64>> = None;
    let mut q_out_prev: Option<UnitQuaternion<f64>> = None;

    let mut yaw_bias: f64 = 0.0;
    let yaw_gain = 0.08;
    let yaw_limit = 30.0_f64.to_radians();

    let mut flex_filter = OneEuroFilter::new(2.0, 0.02, 1.0);
    let mut radial_filter = OneEuroFilter::new(2.0, 0.02, 1.0);
    let mut pinch_filter = OneEuroFilter::new(1.5, 0.01, 1.0);

    let mut pinch_prev = 0.0;
    let pinch_deadband = 0.05;
    let speed_scale = args.speed_scale.clamp(0.01, 1.0);
    let pinch_rate = 0.8 * speed_scale;

    let mut last_cam_for_speed: Option<(UnitQuaternion<f64>, Instant)> = None;
    let mut recenter_requested = false;

    let mut tick = Instant::now();
    let dt_target = 1.0 / args.control_hz.max(1.0);

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        let loop_start = Instant::now();
        drain_imu(&imu_rx, &mut last_imu);
        drain_cam(&cam_socket, &mut last_cam);
        drain_robot(&robot_socket, &mut last_robot);
        while let Ok(()) = ctrl_rx.try_recv() {
            recenter_requested = true;
        }

        let now = loop_start;
        let dt = (now - tick).as_secs_f64().max(1e-4).min(0.1);
        tick = now;


        let imu = match &last_imu {
            Some(sample) if (now - sample.t_recv).as_secs_f64() < 0.05 => sample.clone(),
            _ => {
                sleep_remaining(loop_start, dt_target);
                continue;
            }
        };

        let mut q_e_s = imu.q_e_s.clone();
        if let Some(prev) = &q_imu_smooth {
            if quat_dot(prev, &q_e_s) < 0.0 {
                q_e_s = quat_negate(q_e_s);
            }
            q_e_s = prev.slerp(&q_e_s, 0.15);
        }
        q_imu_smooth = Some(q_e_s.clone());

        let cam_valid = last_cam
            .as_ref()
            .map(|c| c.valid && c.conf > 0.5)
            .unwrap_or(false);
        let cam_pinch_valid = last_cam
            .as_ref()
            .map(|c| c.pinch_valid && c.conf > 0.3)
            .unwrap_or(false);
        let cam_fresh = last_cam
            .as_ref()
            .map(|c| (now - c.t_recv).as_secs_f64() < 0.2)
            .unwrap_or(false);

        if cam_valid && cam_fresh {
            let cam = last_cam.as_ref().unwrap();
            let q_c_h = cam.q_c_h.clone();
            let f_cam = q_c_h.transform_vector(&Vector3::new(0.0, 1.0, 0.0));

            let q_e_f = q_e_s.clone() * q_s_h.clone();
            let f_e = q_e_f.transform_vector(&Vector3::new(0.0, 1.0, 0.0));
            let f_pred = q_c_e.transform_vector(&f_e);

            let theta_cam = f_cam.x.atan2(-f_cam.y);
            let theta_pred = f_pred.x.atan2(-f_pred.y);
            let err = wrap_pi(theta_cam - theta_pred);

            let ang_speed_ok = match last_cam_for_speed {
                Some((ref q_prev, t_prev)) => {
                    let dt_cam = (now - t_prev).as_secs_f64().max(1e-3);
                    let q_delta = q_prev.inverse() * q_c_h.clone();
                    let angle = q_delta.angle();
                    angle / dt_cam < 150.0_f64.to_radians()
                }
                None => true,
            };

            if cam.conf > 0.8 && err.abs() < 45.0_f64.to_radians() && ang_speed_ok {
                yaw_bias += yaw_gain * err * dt;
                yaw_bias = yaw_bias.clamp(-yaw_limit, yaw_limit);
            }

            last_cam_for_speed = Some((q_c_h, now));
        }

        let q_yaw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw_bias);
        let q_e_s_corr = q_yaw * q_e_s.clone();

        let q_e_f = q_e_s_corr.clone() * q_s_h.clone();

        let q_f_h = if cam_valid && cam_fresh {
            if let Some(cam) = &last_cam {
                let q_c_h = cam.q_c_h.clone();
                let q_c_f = q_c_e.clone() * q_e_f.clone();
                let q_f_h_raw = q_c_f.inverse() * q_c_h;

                let y_h = q_f_h_raw.transform_vector(&Vector3::new(0.0, 1.0, 0.0));
                let flex = (-y_h.z).atan2(y_h.y).clamp(-1.2, 1.2);
                let radial = y_h.x.atan2(y_h.y).clamp(-1.2, 1.2);

                let flex_f = flex_filter.filter(flex, now);
                let radial_f = radial_filter.filter(radial, now);

                let q_flex = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), flex_f);
                let q_radial = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), radial_f);
                let q = q_radial * q_flex;

                if let Some(prev) = &q_f_h_smooth {
                    prev.slerp(&q, 0.1)
                } else {
                    q
                }
            } else {
                UnitQuaternion::identity()
            }
        } else {
            q_f_h_smooth.clone().unwrap_or_else(UnitQuaternion::identity)
        };
        q_f_h_smooth = Some(q_f_h.clone());

        let q_e_h = q_e_f * q_f_h;

        if recenter_requested {
            let mut updated = false;
            if let Some(robot) = &last_robot {
                if (now - robot.t_recv).as_secs_f64() < 0.5 {
                    q_b_t_ref = robot.q_b_t.clone();
                    updated = true;
                }
            } else if let Some(prev) = &q_out_prev {
                q_b_t_ref = prev.clone();
                updated = true;
            }
            if updated {
                q_e_h_ref = q_e_h.clone();
                println!("Recenter: updated reference to current hand/robot pose.");
            } else {
                println!("Recenter skipped: no robot pose yet.");
            }
            recenter_requested = false;
        }

        let q_rel = q_e_h_ref.inverse() * q_e_h;
        let mut q_b_t = q_b_t_ref.clone() * q_rel;

        if let Some(prev) = &q_out_prev {
            let angle = (prev.inverse() * q_b_t.clone()).angle();
            let max_step = 120.0_f64.to_radians() * speed_scale * dt;
            if angle > max_step {
                let t = (max_step / angle).clamp(0.0, 1.0);
                q_b_t = prev.slerp(&q_b_t, t);
            }
            if quat_dot(prev, &q_b_t) < 0.0 {
                q_b_t = quat_negate(q_b_t);
            }
        }
        q_out_prev = Some(q_b_t.clone());

        let rotvec = q_b_t.scaled_axis();

        let pinch_norm = if cam_pinch_valid && cam_fresh {
            let cam = last_cam.as_ref().unwrap();
            let raw = cam.pinch;
            let mut p = (raw - calib.pinch_close)
                / (calib.pinch_open - calib.pinch_close).max(1e-6);
            if calib.pinch_open < calib.pinch_close {
                p = (raw - calib.pinch_open) / (calib.pinch_close - calib.pinch_open).max(1e-6);
            }
            p = p.clamp(0.0, 1.0);
            let mut p = pinch_filter.filter(p, now);
            if (p - pinch_prev).abs() < pinch_deadband {
                p = pinch_prev;
            }
            let max_delta = pinch_rate * dt;
            p = p.clamp(pinch_prev - max_delta, pinch_prev + max_delta);
            pinch_prev = p;
            p
        } else {
            pinch_prev
        };
        print("gripper")
        let cmd = CommandMessage {
            kind: "cmd",
            t: time::SystemTime::now()
                .duration_since(time::SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs_f64(),
            ee_pos,
            ee_rotvec: [rotvec.x, rotvec.y, rotvec.z],
            gripper: pinch_norm * 100.0,
        };

        if !args.dry_run {
            let payload = serde_json::to_vec(&cmd)?;
            let _ = cmd_socket.send(&payload);
        }

        sleep_remaining(loop_start, dt_target);
    }

    let _ = terminal::disable_raw_mode();
    Ok(())
}

fn spawn_imu_thread(port: &str, baud: u32, tx: crossbeam_channel::Sender<ImuSample>) -> Result<()> {
    let port = port.to_string();
    thread::spawn(move || {
        if let Err(err) = imu_thread(&port, baud, tx) {
            eprintln!("imu thread error: {err}");
        }
    });
    Ok(())
}

fn spawn_input_thread(tx: crossbeam_channel::Sender<()>) -> Result<()> {
    thread::spawn(move || {
        if terminal::enable_raw_mode().is_err() {
            return;
        }
        loop {
            if crossterm::event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = crossterm::event::read() {
                    if matches!(key.code, KeyCode::Char('x') | KeyCode::Char('X')) {
                        let _ = tx.send(());
                    }
                }
            }
        }
    });
    Ok(())
}

fn imu_thread(port: &str, baud: u32, tx: crossbeam_channel::Sender<ImuSample>) -> Result<()> {
    let mut port: Box<dyn SerialPort> = serialport::new(port, baud)
        .timeout(Duration::from_millis(50))
        .open()
        .with_context(|| format!("open serial port {port}"))?;

    let mut buf = [0u8; 1024];
    let mut pending: Vec<u8> = Vec::new();

    loop {
        match port.read(&mut buf) {
            Ok(0) => continue,
            Ok(n) => {
                pending.extend_from_slice(&buf[..n]);
                while let Some(pos) = pending.iter().position(|b| *b == b'\n') {
                    let line_bytes: Vec<u8> = pending.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line_bytes);
                    let line = line.trim_matches(|c: char| c.is_ascii_whitespace());
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(msg) = serde_json::from_str::<ImuJson>(line) {
                        let q = msg.q;
                        let norm =
                            (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                        if norm < 0.9 {
                            continue;
                        }
                        let q = unit_quat_from_wxyz([q[0], q[1], q[2], q[3]]);
                        let sample = ImuSample {
                            t_recv: Instant::now(),
                            q_e_s: q,
                        };
                        let _ = tx.send(sample);
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(err) => return Err(err.into()),
        }
    }
}

fn drain_imu(rx: &Receiver<ImuSample>, latest: &mut Option<ImuSample>) {
    while let Ok(s) = rx.try_recv() {
        *latest = Some(s);
    }
}

fn drain_cam(socket: &UdpSocket, latest: &mut Option<CamSample>) {
    let mut buf = [0u8; 4096];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((n, _addr)) => {
                if let Ok(msg) = serde_json::from_slice::<HandMessage>(&buf[..n]) {
                    if msg.kind.as_deref() != Some("hand") {
                        continue;
                    }
                    let frame_valid = msg.frame_valid.or(msg.valid).unwrap_or(false);
                    let pinch_valid = msg.pinch_valid.unwrap_or(frame_valid);
                    let conf = msg.conf.unwrap_or(0.0);
                    let pinch = msg.pinch.unwrap_or(0.0);
                    if !frame_valid {
                        *latest = Some(CamSample {
                            t_recv: Instant::now(),
                            conf,
                            q_c_h: UnitQuaternion::identity(),
                            pinch,
                            valid: false,
                            pinch_valid,
                        });
                        continue;
                    }
                    let (x, y, z) = match (msg.x, msg.y, msg.z) {
                        (Some(x), Some(y), Some(z)) => (x, y, z),
                        _ => continue,
                    };
                    if let Some(q_c_h) = axes_to_quat(x, y, z) {
                        *latest = Some(CamSample {
                            t_recv: Instant::now(),
                            conf,
                            q_c_h,
                            pinch,
                            valid: true,
                            pinch_valid,
                        });
                    } else {
                        *latest = Some(CamSample {
                            t_recv: Instant::now(),
                            conf,
                            q_c_h: UnitQuaternion::identity(),
                            pinch,
                            valid: false,
                            pinch_valid,
                        });
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(_) => break,
        }
    }
}

fn drain_robot(socket: &UdpSocket, latest: &mut Option<RobotState>) {
    let mut buf = [0u8; 4096];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((n, _addr)) => {
                if let Ok(msg) = serde_json::from_slice::<RobotStateMessage>(&buf[..n]) {
                    if msg.kind.as_deref() != Some("robot_state") {
                        continue;
                    }
                    let ee_pos = match msg.ee_pos {
                        Some(v) => v,
                        None => continue,
                    };
                    let ee_quat = match msg.ee_quat {
                        Some(v) => v,
                        None => continue,
                    };
                    let q_b_t = unit_quat_from_wxyz(ee_quat);
                    *latest = Some(RobotState {
                        t_recv: Instant::now(),
                        ee_pos,
                        q_b_t,
                    });
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(_) => break,
        }
    }
}

fn sleep_remaining(start: Instant, dt_target: f64) {
    let elapsed = start.elapsed().as_secs_f64();
    let remaining = (dt_target - elapsed).max(0.0);
    thread::sleep(Duration::from_secs_f64(remaining));
}

fn load_calibration(path: &PathBuf) -> Option<Calibration> {
    if !path.exists() {
        return None;
    }
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_calibration(path: &PathBuf, calib: &Calibration) -> Result<()> {
    let data = serde_json::to_string_pretty(calib)?;
    fs::write(path, data)?;
    Ok(())
}

fn run_calibration(
    imu_rx: &Receiver<ImuSample>,
    cam_socket: &UdpSocket,
    robot_socket: &UdpSocket,
    mut calib: Calibration,
) -> Result<Calibration> {
    println!("Calibration starting. Ensure camera and robot bridge are running.");
    println!("Pose 1: forearm forward, palm down, wrist neutral.");
    println!("Align your hand to match the robot tool orientation (robot can stay locked), then press ENTER.");
    wait_enter();

    let pose1 = capture_pose(imu_rx, cam_socket, Duration::from_secs(2))?;
    let robot_state = wait_robot_state(robot_socket, Duration::from_secs(2))?;

    println!("Pose 2: forearm forward, palm inward (thumb up). Press ENTER.");
    wait_enter();
    let pose2 = capture_pose(imu_rx, cam_socket, Duration::from_secs(2))?;

    println!("Pose 3 (optional): forearm to the right, palm down. Press ENTER to capture or type 's' to skip.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let use_pose3 = input.trim().is_empty();
    let pose3 = if use_pose3 {
        Some(capture_pose(imu_rx, cam_socket, Duration::from_secs(2))?)
    } else {
        None
    };

    let mut poses = vec![pose1.clone(), pose2.clone()];
    if let Some(p3) = pose3.clone() {
        poses.push(p3);
    }

    let mut q_s_h = nominal_q_s_h();
    let mut q_c_e = UnitQuaternion::identity();

    for _ in 0..2 {
        let mut q_c_e_list = Vec::new();
        for p in &poses {
            let q_c_h = p.q_c_h.clone();
            let q_e_s = p.q_e_s.clone();
            let q_c_e_i = q_c_h * (q_e_s.clone() * q_s_h.clone()).inverse();
            q_c_e_list.push(q_c_e_i);
        }
        q_c_e = quat_avg(&q_c_e_list).unwrap_or_else(UnitQuaternion::identity);

        let mut q_s_h_list = Vec::new();
        for p in &poses {
            let q_c_h = p.q_c_h.clone();
            let q_e_s = p.q_e_s.clone();
            let q_s_h_i = q_e_s.inverse() * (q_c_e.inverse() * q_c_h);
            q_s_h_list.push(q_s_h_i);
        }
        q_s_h = quat_avg(&q_s_h_list).unwrap_or_else(UnitQuaternion::identity);
    }

    let q_e_h_ref = pose1.q_e_s.clone() * q_s_h.clone();

    println!("Open hand fully for pinch calibration, then press ENTER.");
    wait_enter();
    let open = capture_pinch(cam_socket, Duration::from_secs(2))?;

    println!("Pinch closed for gripper close, then press ENTER.");
    wait_enter();
    let close = capture_pinch(cam_socket, Duration::from_secs(2))?;

    calib.q_s_h = quat_to_wxyz(&q_s_h);
    calib.q_c_e = quat_to_wxyz(&q_c_e);
    calib.q_e_h_ref = quat_to_wxyz(&q_e_h_ref);
    calib.q_b_t_ref = quat_to_wxyz(&robot_state.q_b_t);
    calib.ee_pos = robot_state.ee_pos;
    calib.pinch_open = open;
    calib.pinch_close = close;

    println!("Calibration complete.");
    Ok(calib)
}

#[derive(Clone)]
struct PoseCapture {
    q_e_s: UnitQuaternion<f64>,
    q_c_h: UnitQuaternion<f64>,
}

fn capture_pose(
    imu_rx: &Receiver<ImuSample>,
    cam_socket: &UdpSocket,
    duration: Duration,
) -> Result<PoseCapture> {
    let start_wait = Instant::now();
    let wait_timeout = Duration::from_secs(5);
    let mut last_cam: Option<CamSample> = None;
    while Instant::now() - start_wait < wait_timeout {
        drain_cam(cam_socket, &mut last_cam);
        if let Some(cam) = &last_cam {
            if cam.valid && cam.conf > 0.3 {
                break;
            }
        }
        thread::sleep(Duration::from_millis(10));
    }

    let start = Instant::now();
    let mut imu_samples: Vec<UnitQuaternion<f64>> = Vec::new();
    let mut cam_samples: Vec<UnitQuaternion<f64>> = Vec::new();

    while Instant::now() - start < duration {
        drain_cam(cam_socket, &mut last_cam);
        while let Ok(sample) = imu_rx.try_recv() {
            imu_samples.push(sample.q_e_s);
        }
        if let Some(cam) = &last_cam {
            if cam.valid && cam.conf > 0.3 {
                cam_samples.push(cam.q_c_h.clone());
            }
        }
        thread::sleep(Duration::from_millis(5));
    }

    let q_e_s = quat_avg(&imu_samples).ok_or_else(|| anyhow!("no imu samples"))?;
    let q_c_h = quat_avg(&cam_samples).ok_or_else(|| anyhow!("no camera samples"))?;

    Ok(PoseCapture { q_e_s, q_c_h })
}

fn wait_robot_state(socket: &UdpSocket, timeout: Duration) -> Result<RobotState> {
    let start = Instant::now();
    let mut latest: Option<RobotState> = None;
    while Instant::now() - start < timeout {
        drain_robot(socket, &mut latest);
        if let Some(state) = &latest {
            return Ok(state.clone());
        }
        thread::sleep(Duration::from_millis(10));
    }
    Err(anyhow!("no robot state received"))
}

fn capture_pinch(socket: &UdpSocket, duration: Duration) -> Result<f64> {
    let start = Instant::now();
    let mut values = Vec::new();
    let mut last_cam: Option<CamSample> = None;
    while Instant::now() - start < duration {
        drain_cam(socket, &mut last_cam);
        if let Some(cam) = &last_cam {
            if cam.pinch_valid && cam.conf > 0.3 {
                values.push(cam.pinch);
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
    if values.is_empty() {
        return Err(anyhow!("no pinch samples"));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(values[values.len() / 2])
}

fn wait_enter() {
    let mut input = String::new();
    let _ = std::io::stdin().read_line(&mut input);
}

fn axes_to_quat(x: [f64; 3], y: [f64; 3], _z: [f64; 3]) -> Option<UnitQuaternion<f64>> {
    let mut x = vec3(x);
    let mut y = vec3(y);
    if x.norm() < 1e-6 || y.norm() < 1e-6 {
        return None;
    }
    x = x.normalize();
    y = y.normalize();
    let mut z = x.cross(&y);
    if z.norm() < 1e-6 {
        return None;
    }
    z = z.normalize();
    let cam_z = Vector3::new(0.0, 0.0, 1.0);
    if z.dot(&cam_z) < 0.0 {
        z = -z;
    }
    let y = z.cross(&x).normalize();
    let rot = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, y, z]));
    Some(UnitQuaternion::from_rotation_matrix(&rot))
}

fn vec3(v: [f64; 3]) -> Vector3<f64> {
    Vector3::new(v[0], v[1], v[2])
}

fn nominal_q_s_h() -> UnitQuaternion<f64> {
    let h_x = Vector3::new(0.0, 1.0, 0.0);
    let h_y = Vector3::new(-1.0, 0.0, 0.0);
    let h_z = Vector3::new(0.0, 0.0, 1.0);
    let rot = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[h_x, h_y, h_z]));
    UnitQuaternion::from_rotation_matrix(&rot)
}

fn quat_to_wxyz(q: &UnitQuaternion<f64>) -> [f64; 4] {
    let inner = q.clone().into_inner();
    [inner.w, inner.i, inner.j, inner.k]
}

fn unit_quat_from_wxyz(q: [f64; 4]) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(q[0], q[1], q[2], q[3]))
}

fn quat_avg(quats: &[UnitQuaternion<f64>]) -> Option<UnitQuaternion<f64>> {
    if quats.is_empty() {
        return None;
    }
    let mut a = Matrix4::zeros();
    let ref_q = quats[0].clone();
    for q in quats {
        let mut qn = q.clone();
        if quat_dot(&ref_q, &qn) < 0.0 {
            qn = quat_negate(qn);
        }
        let v = quat_vec4(&qn);
        a += v * v.transpose();
    }
    let eig = SymmetricEigen::new(a);
    let mut max_i = 0;
    for i in 1..4 {
        if eig.eigenvalues[i] > eig.eigenvalues[max_i] {
            max_i = i;
        }
    }
    let v = eig.eigenvectors.column(max_i);
    let q = Quaternion::new(v[0], v[1], v[2], v[3]);
    Some(UnitQuaternion::new_normalize(q))
}

fn quat_vec4(q: &UnitQuaternion<f64>) -> Vector4<f64> {
    let inner = q.clone().into_inner();
    Vector4::new(inner.w, inner.i, inner.j, inner.k)
}

fn quat_dot(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
    let ai = a.clone().into_inner();
    let bi = b.clone().into_inner();
    ai.w * bi.w + ai.i * bi.i + ai.j * bi.j + ai.k * bi.k
}

fn quat_negate(q: UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(-q.into_inner())
}

fn wrap_pi(mut x: f64) -> f64 {
    while x > std::f64::consts::PI {
        x -= 2.0 * std::f64::consts::PI;
    }
    while x < -std::f64::consts::PI {
        x += 2.0 * std::f64::consts::PI;
    }
    x
}
