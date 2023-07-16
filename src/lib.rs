use config::Config;

use frontend::run;
use git::git_stuff;

use uuid::Uuid;

mod card;
mod categories;
mod cli;
mod common;
mod config;
mod frontend;
mod git;
mod media;
mod ml;

pub mod paths {
    use std::path::PathBuf;

    pub fn get_import_csv() -> PathBuf {
        get_share_path().join("import.csv")
    }

    pub fn get_cards_path() -> PathBuf {
        get_share_path().join("cards")
    }

    pub fn get_ml_path() -> PathBuf {
        get_share_path().join("ml/")
    }

    pub fn get_runmodel_path() -> PathBuf {
        get_ml_path().join("runmodel.py")
    }

    pub fn get_media_path() -> PathBuf {
        get_share_path().join("media/")
    }

    #[cfg(not(test))]
    pub fn get_share_path() -> PathBuf {
        let home = dirs::home_dir().unwrap();
        home.join(".local/share/speki/")
    }

    #[cfg(test)]
    pub fn get_share_path() -> PathBuf {
        PathBuf::from("./test_dir/")
    }
}

type Id = Uuid;

fn main() {
    std::fs::create_dir_all(paths::get_cards_path()).unwrap();
    std::fs::create_dir_all(paths::get_share_path().join("media/")).unwrap();

    let config = Config::load().unwrap();
    std::thread::spawn(move || git_stuff(config.read_git_remote()));
    run();
}
