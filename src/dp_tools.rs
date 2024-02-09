//! 動的計画法を用いた計算用ツール集

pub mod calc_dp;
pub mod calc_dp_2;


/// `cpd_tools::calc_dp`に関するError
#[derive(Debug, Clone)]
pub struct CalcDpError {
    pub message: String,
}

impl std::fmt::Display for CalcDpError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CalcDpError {
    fn description(&self) -> &str {
        &self.message
    }
}
