

use std::result::Result;
use std::error::Error;
use std::fmt::{self, Formatter, Display, Debug};


pub type CumathResult<T> = Result<T, CumathError>;


pub struct CumathError {
    description: &'static str,
}
impl Display for CumathError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}
impl Debug for CumathError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}
impl Error for CumathError {
    fn description(&self) -> &str {
        self.description
    }
    fn cause(&self) -> Option<&Error> {
        None
    }
}
impl CumathError {
    pub fn new(description: &'static str) -> CumathError {
        CumathError { description }
    }
}