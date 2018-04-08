
use std::path::Path;
use std::io::Read;
use std::io::Write;
use std::fs::File;
use std::fs::OpenOptions;

use super::codec::*;


/// Save behaviour, used to save data.
pub trait Save: Codec {
    fn save(&self, file_path: &str);
    fn load(file_path: &str) -> <Self as Codec>::OutputType;
}

impl<T: Codec> Save for T {
    fn save(&self, file_path: &str) {
        let path = Path::new(file_path);
        let mut file = OpenOptions::new().create(true).write(true)
            .open(path).expect("Couldn't open file");
        file.set_len(0).expect("Couldn't clear file");
        file.write(self.encode().as_bytes()).expect("Couldn't write to file");
    }
    fn load(file_path: &str) -> <T as Codec>::OutputType {
        let path = Path::new(file_path);
        let mut file : File = File::open(path).expect("Couln't open file");
        let mut data = String::new();
        file.read_to_string(&mut data).expect("Couldn't read file");
        Self::decode(data.as_str())
    }
}