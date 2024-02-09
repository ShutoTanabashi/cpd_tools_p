//! 最低間隔が2個の場合の動的計画法(DP)を用いた評価値計算のためのプログラム集
//! 
//! # 想定する問題
//! 2個の連続した変化点$ t_k, t_{k-1} $の間にデータが2個以上，すなわち$ t_k - 1 > t_{k-1}$であるとする．
//! そのうえで変化点$ t_k, t_{k-1} $が与えられたとき，データ$ \bm{X} $から評価値を計算する関数$ f(t_k, t_{k-1} | \bm{X}) $が定義される場合を想定．
//! 更に，データ全体に対する評価値が各変化点間の評価値の総和$ \sum_{k=1}^{K} f(t_k, t_{k-1}) $を利用して計算される場合も扱う．

use super::CalcDpError;

extern crate rayon;
use rayon::prelude::*;

extern crate process_param;
use process_param::{Tau, NumChg};


/// 変化点の順序を確認する
///
/// # 引数
/// * `t_k_1` - 前の変化点 $t_{k-1}$
/// * `t_k` - 後ろの変化点 $t_k$
pub fn order_change_point(t_k_1: &Tau, t_k: &Tau) -> Result<(), CalcDpError> {
    if *t_k == 0 {
        Err( CalcDpError{
            message: format!("Index tau_{{k}} (={t_k}) must be greater than 0.")
        })
    } else if (*t_k_1 >= (*t_k - 1)) && !(*t_k == 1 && *t_k_1 == 0) {
        Err( CalcDpError{
            message: format!("Index tau_{{k}} (={t_k}) must be greater than tau_{{k-1}} + 1 (= {t_k_1}+1)")
        })
    } else {
        Ok(())
    }
}


/// 2つの変化点間における計算が可能
pub trait CalcTT<Val, Ipt> where
{
    /// 2個の変化点間の評価値を計算する関数$ f(t_k, t_{k-1} | \bm{X}) $
    ///
    /// # 引数
    /// * `data` - 計算に用いるデータ$ \bm{X} $
    /// * `t_k_1` - 前の変化点 $t_{k-1}$
    /// * `t_k` - 後ろの変化点 $t_k$
    fn calc_value(data: &Ipt, t_k_1: Tau, t_k: Tau) -> Result<Val, CalcDpError>;
}


/// 2つの変化点間における計算結果を格納する
/// 
/// # 利用するジェネリクス型
/// * `Val` - 計算結果の値の型
/// * `Ipt` - 計算に用いるデータの型
pub trait DictTT<Val, Ipt>: CalcTT<Val, Ipt> where
    Val: Clone + std::marker::Send + std::fmt::Debug,
    Ipt: std::marker::Sync
{
    /// 任意の2個の変化点間の値を格納した2次元配列
    /// 
    /// # 関数制作時の注意
    /// 返り値となる2次元配列についてですが，1個目の要素数が変化点，2個目の要素数が変化点からの経過時間です．
    /// ただし，変化点はデータが切り替わる直前の時点として定義されることに注意してください．
    /// 例えば，2個の連続する変化点$ t_k, t_{k-1} $に対してその間の値$ f(t_k, t_{k-1}) $を得る場合，スライスのインデックスは`[t_{k-1}][t_k - (t_{k-1} + 1)]`となります．
    fn value_tt_all(&self) -> Vec<Vec<Val>>;

    /// 任意の2個の変化点間の値を返す
    ///
    /// # 引数
    /// * `t_k_1` - 前の変化点 $t_{k-1}$
    /// * `t_k` - 後ろの変化点 $t_k$
    fn value_tt(&self, t_k_1: Tau, t_k: Tau) -> Result<Val, CalcDpError> {
        order_change_point(&t_k_1, &t_k)?;

        // 1個目の変化点確認
        let vals_all = self.value_tt_all();
        let vals_tau_k_1 = if vals_all.len() < (t_k_1 as usize) {
                return Err( CalcDpError{
                    message: format!("Index tau_{{k - 1}} (={t_k_1}) is out of range.")
                })
            } else {
                &vals_all[t_k_1 as usize]
            };

        // 2個目の変化点確認
        let index_tt = t_k - t_k_1 - 2;
        if vals_tau_k_1.len() < (index_tt as usize) {
            Err( CalcDpError{
                message: format!("Index tau_{{k}} (={t_k}) must be greater than tau_{{k-1}} + 1 (={t_k_1} + 1)")
            })
        } else {
            Ok(vals_tau_k_1[index_tt as usize].clone())
        }
    }


    /// 2個の変化点間の評価値を格納した2次元配列を作成
    ///
    /// # 引数
    /// * `data` - 計算に用いるデータ$ \bm{X} $
    /// * `t_max` - 変化点の最大値（最後の時期）
    ///
    /// # 返り値
    /// * `vals` - 評価値を格納した2次元配列．
    ///
    /// ## 返り値の構造について
    /// 配列のインデックスについては，1個目の要素数が変化点，2個目の要素数が変化点からの経過時間を示す．ただし，変化点はデータが切り替わる直前の時点として定義される．
    /// 例えば，2個の連続する変化点$ t_k, t_{k-1} $に対してその間の値$ f(t_k, t_{k-1}) $を得る場合，スライスのインデックスは`[t_{k-1}][t_k - (t_{k-1} + 1)]`となる．
    fn calc_value_all(data: &Ipt, t_max: &Tau) -> Result<Vec<Vec<Val>>, CalcDpError> {
        (0..(*t_max-1)).into_par_iter()
                   .map(
                       |t_k_1| ((t_k_1 + 2)..=*t_max).map(
                           |t_k| Self::calc_value(data, t_k_1, t_k)
                                                    ).collect()
                  ).collect()
    }
}


/// DictTTを利用して，任意の変化点群に対する評価関数を計算する
///
/// 主に動的計画法が用いれないため全探索を行う場合での利用を想定．
pub trait DictToFunc<'a, Val, Ipt>: DictTT<Val, Ipt> where
    Val: std::iter::Sum + Clone + std::marker::Send + std::fmt::Debug,
    Ipt: std::marker::Sync
{
    /// 変化点群から評価関数の値を返す
    ///
    /// # 引数
    /// * `change_points` - 計算対象の変化点群
    fn evaluate(&self, change_points: &[Tau]) -> Result<Val, CalcDpError>;
    

    /// 変化点群から評価値の合計を計算する
    /// 
    /// DictTTに格納された値の合計値であり，関数の計算結果であるとは限らない．
    /// 関数の計算結果が欲しい場合は `evaluate` メソッドを利用すること．
    ///
    /// # 引数
    /// * `change_points` - 計算対象の変化点群
    fn sum_frol_cp(&self, change_points: &[Tau]) -> Result<Val, CalcDpError> {
        // イテレータを用意
        let mut cp_copy = change_points.to_vec();
        cp_copy.insert(0,0);
        cp_copy.pop();
        let comb_cp = cp_copy.iter().zip(change_points.iter());
        
        let vals_tt = comb_cp.map(|(t_k_1, t_k)| self.value_tt(*t_k_1, *t_k))
                             .collect::<Result<Vec<Val>, CalcDpError>>()?;
        let val = vals_tt.into_iter()
                         .sum();
        Ok(val)
    }
}


/// 動的計画法で評価値を計算する
///
/// # 計算に用いるメモについて
/// ([`Tau`], [`NumChg`], `Val`)を要素とする2次元ベクトル．
/// 順に(`一つ前の期数`, `現在の変化点個数`, `現時点での評価値`)で成り立つ．
/// 2次元ベクトルの各軸については，1次元目が変化点個数，2次元目が時期である．
pub trait CalcDP<Val, Ipt>: CalcTT<Val, Ipt> where
    Val: std::iter::Sum + std::cmp::PartialOrd + Clone + std::fmt::Debug,
{
    /// 動的計画法によりすべての評価値を格納したメモを作成
    ///
    /// # 引数
    /// * `data` - 計算に必要な入力値
    /// * `t_max` - 変化点の最大値（最後の時期）
    fn calc_memo_all(data: &Ipt, t_max: &Tau) -> Result<Vec<Vec<Option<(Tau, NumChg, Val)>>>, CalcDpError> {
        let k_max = Self::calc_max_k(t_max);
        let mut memo = (0..=k_max).map(|i| vec![None; (t_max - (2 * i) + 1) as usize] )
                                  .collect::<Vec<Vec<Option<(Tau, NumChg, Val)>>>>();
        
        // メモを計算
        for k in 0..=k_max { 
            Self::calc_memo(t_max, &k, &mut memo, data)?;
        };
       
        Ok(memo)
    }


    /// 動的計画法の計算に用いたメモを返す
    ///
    /// # 注意
    /// [`Self::calc_memo_all`]の返り値を返してください．
    /// 計算コストを考慮して，`struct`の要素としてメモを保持する状況を想定しています．
    fn memo_all(&self) -> Vec<Vec<Option<(Tau, NumChg, Val)>>>;


    /// 評価値の推移を取得
    ///
    /// 指定された変化点と変化回数から，その評価値等を計算に用いた中間地点の評価値等とともに出力する．
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    fn get_value_history(&self, t: &Tau, k: &NumChg) -> Result<Vec<(Tau, NumChg, Val)>, CalcDpError> {
        let mut now_t = *t;
        let mut now_k = *k;
        let memo = self.memo_all();
        let mut res = Vec::new();

        while now_t > 0 {
            let memo_tk;
            match Self::get_from_memo(&now_t, &now_k, &memo)? {
                None => {
                    // 値が設定されていない場合はエラーとなる．
                    return Err(CalcDpError{
                        message: "Uncalculated value exist.".to_owned()
                    });
                },
                Some(v) => memo_tk = v,
            };
            
            now_t = memo_tk.0;
            if memo_tk.1 != 0 {
                now_k = memo_tk.1 - 1;
            };
            res.push(memo_tk);
        }
        Ok(res)
    }


    /// 評価値を取得
    ///
    /// 指定された変化点と変化回数の評価値を返す．
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    fn get_value(&self, t: &Tau, k: &NumChg) -> Result<Val, CalcDpError> {
        match Self::get_from_memo(t, k, &self.memo_all())? {
            Some(v) => Ok(v.2),
            None => Err(CalcDpError{
                message: "Value has not calculated yet.".to_owned()
            }),
        }
    }


    /// memoに対してインデックスtおよびkが正しいか確認
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    /// * `memo` - 動的計画法の計算に用いるメモ
    fn check_idx_memo(t: &Tau, k: &NumChg, memo: &[Vec<Option<(Tau, NumChg, Val)>>]) -> Result<(), CalcDpError> {
        if (*t as usize) > (memo[0].len() - 1) {
            return Err(CalcDpError{
                message: format!("Time step t = {t} is out of range.")
            });
        }

        if *t == 0 {
            return Err(CalcDpError{
                message: "Time step must be greater than 0".to_owned()
            });
        }

        let max_k = Self::calc_max_k(t);
        if *k > max_k {
            return Err(CalcDpError{
                message: format!("The number of change point k (= {k}) must be less than ceil( (t-1)/2 ) (= {max_k}).")
            });
        }

        Ok(())
    }


    /// メモから値を取得
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    /// * `memo` - 動的計画法の計算に用いるメモ
    fn get_from_memo(t: &Tau, k: &NumChg, memo: &[Vec<Option<(Tau, NumChg, Val)>>]) -> Result<Option<(Tau, NumChg, Val)>, CalcDpError> {
        Self::check_idx_memo(t, k, memo)?;
        Ok( memo[*k as usize][(*t-(*k * 2)) as usize].clone() )
    }


    /// メモに値をセット
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    /// * `eval` - 評価値
    /// * `memo` - 動的計画法の計算に用いるメモ
    fn set_from_memo(t: &Tau, val: (Tau, NumChg, Val), memo: &mut [Vec<Option<(Tau, NumChg, Val)>>]) -> Result<(Tau, NumChg, Val), CalcDpError> {
        let k = val.1;
        Self::check_idx_memo(&t, &k, memo)?;
        memo[k as usize][(t-(k*2)) as usize] = Some(val.clone());
        Ok(val)
    }

    
    /// 動的計画法を用いて評価値を計算する
    ///
    /// # 引数
    /// * `t` - 計算する期数
    /// * `k` - 計算する変化点個数
    /// * `memo` - 動的計画法の計算に用いるメモ
    /// * `data` - 計算に必要な入力値
    fn calc_memo(t: &Tau, k: &NumChg, memo: &mut [Vec<Option<(Tau, NumChg, Val)>>], data: &Ipt) -> Result<(Tau, NumChg, Val), CalcDpError> {
        Self::check_idx_memo(t, k, memo)?;

        // k=0なら再帰の末尾．別処理
        if *k == 0 {
            return match Self::get_from_memo(t, k, memo)? {
                Some(v) => Ok(v),
                None => {
                    let eval = Self::calc_value(data, 0, *t)?;
                    let res_tk = (0, 0, eval);
                    Self::set_from_memo(t, res_tk, memo)
                },
            }
        }

        // k>0の場合
        // ひとつ前の変化点$ \tau_{k-1} $ごとに評価値を計算
        let mut vals = Vec::with_capacity((t - 2 * k + 1) as usize);

        for i in (*k * 2 - 1)..(*t - 1) {
            let max_k_1 = {
                let tpl_mk1 = match Self::get_from_memo(&i, &(*k-1), memo)? {
                    Some(v) => v,
                    None => Self::calc_memo(&i, &(*k-1), memo, data)?,
                };
                tpl_mk1.2
            };
            let val_tt = Self::calc_value(data, i, *t)?;
            let eval:Val = [max_k_1, val_tt].into_iter()
                                            .sum();
            let res_tk = (i, *k, eval);
            vals.push(res_tk);
        } 

        // 評価値最大のものを選択
        let op_max_val = vals.iter()
                             .reduce(|acc, val| {
                                if acc.2 <= val.2 {
                                    val
                                } else {
                                    acc
                                }
                            });
        let max_val;
        match op_max_val {
            Some(v) => max_val = v,
            None => return Err( CalcDpError{
                message: "Failed to compute dynamic programming memo.".to_owned()
            }),
        };
       
        Self::set_from_memo(t, max_val.clone(), memo)
    }


    /// Kの最大値を計算
    ///
    /// # 引数
    /// * `t_max` - 変化点の最大値（最後の時期）
    ///
    /// # 返り値
    /// * `k_max` - 変化点個数の最大値
    fn calc_max_k(t_max: &Tau) -> NumChg {
        // 天井関数の代わりに整数の割り算では余りが切り捨てられることを利用
        ((*t_max - 1) / 2) as NumChg
    }
}
