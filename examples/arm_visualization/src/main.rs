//! An example of how to load and animate a URDF given some changing joint angles.
//!
//! Usage:
//! ```
//! cargo run -p animated_urdf
//! ```

#[derive(Debug, clap::Parser)]
#[clap(author, version, about)]
struct Args {
    #[command(flatten)]
    rerun: rerun::clap::RerunArgs,
}

use rerun::components::Translation3D;
use rerun::external::re_data_loader::UrdfTree;
use rerun::external::{re_log, urdf_rs};

use serde_json::Value as JsonValue;
use std::io::ErrorKind;
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    re_log::setup_logging();

    use clap::Parser as _;
    let args = Args::parse();

    let (rec, _serve_guard) = args.rerun.init("rerun_example_animated_urdf")?;
    run(&rec, &args)
}

fn run(rec: &rerun::RecordingStream, _args: &Args) -> anyhow::Result<()> {
    let urdf_path = "../../rerun_arm/robot.urdf";

    // Log the URDF file one, as a static resource:
    rec.log_file_from_path(urdf_path, None, true)?;

    // Load the URDF tree structure into memory:
    let urdf = UrdfTree::from_file_path(urdf_path)?;

    // Shared map of requested joint angles (radians). JSON input should be a flat map
    // from joint name -> angle in degrees, e.g. {"joint1": 45, "joint2": -30}.
    let joint_commands: Arc<Mutex<std::collections::HashMap<String, f64>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));

    // Spawn UDP listener thread
    {
        let commands = joint_commands.clone();
        thread::spawn(move || {
            // Bind to a UDP port (0.0.0.0:9999)
            let socket = match UdpSocket::bind("0.0.0.0:9999") {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to bind UDP socket: {}", e);
                    return;
                }
            };

            // Non-blocking receive so the thread can sleep periodically
            if let Err(e) = socket.set_nonblocking(true) {
                eprintln!("Failed to set nonblocking UDP socket: {}", e);
                return;
            }

            let mut buf = [0u8; 2048];
            loop {
                match socket.recv_from(&mut buf) {
                    Ok((len, src)) => {
                        let msg = match std::str::from_utf8(&buf[..len]) {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("Invalid UTF-8 in UDP packet from {}: {}", src, e);
                                continue;
                            }
                        };

                        match serde_json::from_str::<JsonValue>(msg) {
                            Ok(JsonValue::Object(map)) => {
                                let mut guard = match commands.lock() {
                                    Ok(g) => g,
                                    Err(_) => {
                                        eprintln!("Joint command mutex poisoned");
                                        continue;
                                    }
                                };

                                for (k, v) in map {
                                    if let Some(deg) = v.as_f64() {
                                        let clamped_deg = deg.clamp(-90.0, 90.0);
                                        let rad = clamped_deg.to_radians();
                                        guard.insert(k.clone(), rad);
                                        println!(
                                            "[udp] cmd {} = {}° ({} rad)",
                                            k, clamped_deg, rad
                                        );
                                    } else {
                                        eprintln!("Value for joint '{}' is not a number", k);
                                    }
                                }
                            }
                            Ok(_) => {
                                eprintln!("Expected JSON object mapping joint->angle in degrees");
                            }
                            Err(e) => {
                                eprintln!("Failed to parse JSON from {}: {}", src, e);
                            }
                        }
                    }
                    Err(e) => {
                        if e.kind() != ErrorKind::WouldBlock {
                            // Non-critical: print and continue
                            eprintln!("UDP recv error: {}", e);
                        }
                        // Sleep a bit to avoid busy loop
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            }
        });
    }

    // Animate continuously and apply commanded joint angles when present:
    let mut step: i64 = 0;
    loop {
        rec.set_time_sequence("step", step);

        for (joint_index, joint) in urdf.joints().enumerate() {
            if joint.joint_type == urdf_rs::JointType::Revolute {
                let fixed_axis = joint.axis.xyz.0;

                // If we have a commanded angle for this joint, use it; otherwise use the fake angle.
                let dynamic_angle = {
                    let guard = joint_commands.lock().unwrap();
                    if let Some(&cmd_rad) = guard.get(&joint.name) {
                        println!("got angle");
                        // Clamp to joint limits too
                        emath::remap(cmd_rad, -90.0..=90.0, joint.limit.lower..=joint.limit.upper)
                    } else {
                        continue;
                    }
                };

                // Plot the joint angle over time - each joint gets its own path for toggling
                rec.log(
                    format!("plots/joint_angles/{}", joint.name),
                    &rerun::Scalars::new([dynamic_angle]),
                )?;

                // Compute the full rotation for this joint.
                let rotation = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    joint.origin.rpy[0] as f32,
                    joint.origin.rpy[1] as f32,
                    joint.origin.rpy[2] as f32,
                ) * glam::Quat::from_axis_angle(
                    glam::Vec3::new(
                        fixed_axis[0] as f32,
                        fixed_axis[1] as f32,
                        fixed_axis[2] as f32,
                    ),
                    dynamic_angle as f32,
                );

                // Rerun loads the URDF transforms with child/parent frame relations.
                // In order to move a joint, we just need to log a new transform between two of those frames.
                rec.log(
                    "/transforms",
                    &rerun::Transform3D::from_rotation(rerun::Quaternion::from_xyzw(
                        rotation.to_array(),
                    ))
                    .with_translation(Translation3D::from(joint.origin.xyz.0))
                    .with_parent_frame(joint.parent.link.clone())
                    .with_child_frame(joint.child.link.clone()),
                )?;
            }
        }

        step += 1;
        // Adjust sleep for update rate (e.g., 20 Hz)
        thread::sleep(Duration::from_millis(50));
    }

    // Ok(()) // unreachable
}
