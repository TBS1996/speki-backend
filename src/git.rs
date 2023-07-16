use std::process::Command;

use crate::paths::get_share_path;

pub fn git_save(has_remote: bool) {
    Command::new("git").args(["add", "."]).output().unwrap();
    Command::new("git")
        .args(["commit", "-m", "save"])
        .output()
        .unwrap();

    if has_remote {
        Command::new("git")
            .args(["push", "-u", "origin", "main"])
            .output()
            .unwrap();
    }
}

pub fn git_pull() {
    Command::new("git").args(["pull"]).output().unwrap();
}

pub fn git_stuff(git_remote: &Option<String>) {
    std::env::set_current_dir(get_share_path()).unwrap();

    // Initiate git
    Command::new("git").arg("init").output().unwrap();

    if let Some(git_remote) = git_remote {
        // Check if the remote repository is already set
        let remote_check_output = Command::new("git")
            .args(["remote", "get-url", "origin"])
            .output()
            .unwrap();

        if remote_check_output.status.success() {
            git_pull();
        } else {
            // Set the remote repository
            Command::new("git")
                .args(["remote", "add", "origin", git_remote])
                .output()
                .unwrap();
        }
    }

    git_save(git_remote.is_some());
}
