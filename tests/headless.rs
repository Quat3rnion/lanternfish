#![cfg(not(feature = "ui"))]

use std::process::Command;

#[test]
fn headless_cli_exports_generated_gguf_without_server_or_store() {
    let bin = env!("CARGO_BIN_EXE_lanternfish");
    let help = Command::new(bin).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(
        !help
            .lines()
            .any(|line| line.trim_start().starts_with("serve ")),
        "{help}"
    );
    let tests = Command::new("python3")
        .args(["-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("LANTERNFISH_BIN", bin)
        .status()
        .expect("headless CLI tests require Python 3");
    assert!(tests.success());
}
