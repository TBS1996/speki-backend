//use rodio::Source;

use serde::{Deserialize, Serialize};

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Clone, Serialize, Debug, Default)]
pub struct AudioSource {
    #[serde(default)]
    #[serde(rename = "audio_local")]
    local_name: Option<String>,
    #[serde(default)]
    #[serde(rename = "audio_url")]
    url_backup: Option<String>,
}

impl AudioSource {
    pub fn new(local_name: Option<String>, url_backup: Option<String>) -> Self {
        Self {
            local_name,
            url_backup,
        }
    }

    /*

    pub fn _play_audio(&mut self) -> Option<std::thread::JoinHandle<()>> {
        if !Config::load().ok()?.play_audio {
            return None;
        }

        let path = self.get_path()?;

        let handle = std::thread::spawn(move || {
            let (_stream, stream_handle) = rodio::OutputStream::try_default().unwrap();

            // Load a sound from a file
            let file = BufReader::new(File::open(path).unwrap());
            let source = rodio::Decoder::new(file).unwrap();

            // Play the sound
            stream_handle.play_raw(source.convert_samples()).unwrap();

            // Block the thread until audio is done.
            std::thread::sleep(std::time::Duration::from_secs(30));
        });

        Some(handle)
    }
    */
}

/*
impl GetMedia for AudioSource {
    fn local_name(&self) -> Option<String> {
        self.local_name.clone()
    }

    fn url_backup(&self) -> Option<String> {
        self.url_backup.clone()
    }

    fn update_local(&mut self, name: &str) {
        self.local_name = Some(name.into());
    }
}

pub trait GetMedia {
    fn local_name(&self) -> Option<String>;
    fn url_backup(&self) -> Option<String>;
    fn update_local(&mut self, name: &str);

    fn local_name_as_path(&self) -> Option<PathBuf> {
        let local_file = self.local_name()?;
        Some(get_media_path().join(local_file))
    }

    fn get_local_path(&self) -> Option<PathBuf> {
        let local_name = self.local_name_as_path()?;
        if local_name.exists() {
            return Some(local_name);
        }
        None
    }

    fn download_media(&mut self) -> Option<PathBuf> {
        let url = self.url_backup()?;

        let response = ureq::get(&url).call().unwrap();

        let fname = url.rsplit('/').next().unwrap();
        let path = get_media_path().join(fname);

        let mut out = std::fs::File::create(&path).unwrap();

        let mut reader = response.into_reader();
        std::io::copy(&mut reader, &mut out).ok()?;

        self.update_local(fname);

        Some(path)
    }

    fn get_path(&mut self) -> Option<PathBuf> {
        self.get_local_path().or_else(|| self.download_media())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_media() {
        let mut media = AudioSource {
            local_name: None,
            url_backup: Some("https://tatoeba.org/en/audio/download/269099".into()),
        };
        let path = media.get_path().unwrap();
        assert_eq!(media.local_name.unwrap(), "269099");
        std::fs::remove_file(path).unwrap();
    }
}
*/
