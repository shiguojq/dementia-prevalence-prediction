import numpy as np
import pandas as pd
import warnings
import os
import glob
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')  # 避免警告信息干扰

# 安全log转换函数
def safe_log(x):
    x = np.where(x <= 0, 0.0001, x)
    x = np.where(x >= 1, 0.9999, x)
    return np.log(x)

# 寻找ARIMA最佳参数的核心函数
def select_best_arima_order(residuals, max_p=3, max_d=2, max_q=3):
    """
    根据AIC准则自动选择最佳ARIMA阶数，优化d值选择过程
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_model = None
    
    # 第一步：确定最优差分阶数d
    optimal_d = 0
    p_values = []
    adf_results = []
    
    for d_test in range(max_d + 1):
        if d_test == 0:
            series_to_test = residuals
        else:
            series_to_test = np.diff(residuals, n=d_test)
        
        try:
            adf_result = adfuller(series_to_test)
            p_value = adf_result[1]
            p_values.append((d_test, p_value))
            adf_results.append({
                'd': d_test,
                'p_value': p_value,
                'adf_statistic': adf_result[0],
                'critical_value_1%': adf_result[4]['1%'],
                'critical_value_5%': adf_result[4]['5%'],
                'critical_value_10%': adf_result[4]['10%']
            })
            
            if p_value < 0.05:
                optimal_d = d_test
                if d_test > 0:
                    break
        except Exception as e:
            print(f"  d={d_test}时ADF检验失败: {e}")
            continue
    
    print(f"选择的差分阶数d: {optimal_d}")
    
    # 第二步：搜索p和q
    candidate_orders = []
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0 and optimal_d == 0:
                continue
                
            current_order = (p, optimal_d, q)
            
            try:
                model = ARIMA(residuals, order=current_order)
                model_fit = model.fit()
                
                # 安全地获取模型属性，兼容不同版本
                candidate_info = {
                    'order': current_order,
                    'p': p,
                    'd': optimal_d,
                    'q': q,
                    'aic': model_fit.aic,
                    'bic': model_fit.bic,
                    'log_likelihood': model_fit.llf,
                    'converged': model_fit.mle_retvals.get('converged', True) if hasattr(model_fit, 'mle_retvals') else True
                }
                
                # 兼容性处理：不同版本的属性名
                if hasattr(model_fit, 'hqic'):
                    candidate_info['hqic'] = model_fit.hqic
                
                # 处理sigma2属性 - 不同版本可能名称不同
                if hasattr(model_fit, 'sigma2'):
                    candidate_info['sigma2'] = model_fit.sigma2
                elif hasattr(model_fit, 'params') and len(model_fit.params) > 0:
                    # 尝试从参数中获取方差估计
                    candidate_info['sigma2'] = model_fit.params[-1] if 'sigma2' in model_fit.param_names else None
                else:
                    candidate_info['sigma2'] = None
                
                # 获取残差信息
                if hasattr(model_fit, 'resid'):
                    candidate_info['residuals_mean'] = np.mean(model_fit.resid)
                    candidate_info['residuals_std'] = np.std(model_fit.resid)
                    candidate_info['residuals_min'] = np.min(model_fit.resid)
                    candidate_info['residuals_max'] = np.max(model_fit.resid)
                else:
                    candidate_info['residuals_mean'] = None
                    candidate_info['residuals_std'] = None
                    candidate_info['residuals_min'] = None
                    candidate_info['residuals_max'] = None
                
                candidate_orders.append(candidate_info)
                
            except Exception as e:
                continue
    
    # 创建候选模型DataFrame并明确按AIC排序
    candidates_df = None
    if candidate_orders:
        candidates_df = pd.DataFrame(candidate_orders)
        # 明确按AIC升序排序，最小的在最前面
        candidates_df = candidates_df.sort_values('aic', ascending=True).reset_index(drop=True)
        
        # 确保候选模型列表也按AIC排序
        candidate_orders = candidates_df.to_dict('records')
    
    # 如果没有找到合适的模型，使用默认值
    if not candidate_orders:
        print("警告：未找到合适的ARIMA模型，使用默认(1,1,1)")
        try:
            model = ARIMA(residuals, order=(1, 1, 1))
            best_model = model.fit()
            best_order = (1, 1, 1)
            best_aic = best_model.aic
            
            # 为默认模型创建候选信息
            default_candidate = {
                'order': (1, 1, 1),
                'p': 1,
                'd': 1,
                'q': 1,
                'aic': best_model.aic,
                'bic': best_model.bic,
                'log_likelihood': best_model.llf,
                'converged': best_model.mle_retvals.get('converged', True) if hasattr(best_model, 'mle_retvals') else True,
                'is_default': True
            }
            
            candidate_orders = [default_candidate]
            candidates_df = pd.DataFrame(candidate_orders)
            
        except Exception as e:
            print(f"默认模型拟合也失败: {e}")
            # 如果连默认模型都失败，返回简单信息
            return (1, 1, 1), None, {
                'adf_results': adf_results,
                'candidate_models': None,
                'optimal_d': optimal_d,
                'residuals_info': {
                    'length': len(residuals),
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals)
                },
                'error': str(e)
            }
    else:
        # 选择AIC最小的模型 - 现在candidate_orders已经按AIC排序
        best_candidate = candidate_orders[0]  # 第一个就是AIC最小的
        best_order = best_candidate['order']
        best_aic = best_candidate['aic']
        
        # 重新拟合最佳模型
        try:
            model = ARIMA(residuals, order=best_order)
            best_model = model.fit()
        except Exception as e:
            print(f"重新拟合最佳模型失败: {e}")
            best_model = None
        
        # 打印前3个最佳模型
        print(f"最佳ARIMA阶数: {best_order}, AIC: {best_aic:.2f}")
        print("前3个候选模型（按AIC从小到大）:")
        for i, cand in enumerate(candidate_orders[:3]):
            hqic_str = f"{cand.get('hqic', 'N/A'):.2f}" if cand.get('hqic') is not None else 'N/A'
            print(f"  {i+1}. ARIMA{cand['order']}: AIC={cand['aic']:.2f}, BIC={cand['bic']:.2f}, HQIC={hqic_str}")
    
    # 返回结果
    return best_order, best_model, {
        'adf_results': adf_results,
        'candidate_models': candidates_df,
        'optimal_d': optimal_d,
        'residuals_info': {
            'length': len(residuals),
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals)
        }
    }

# 用于分析残差并找到最佳ARIMA参数的函数
def find_best_arima_for_residuals(residuals, save_path=None):
    """
    对给定的残差序列寻找最佳ARIMA参数
    
    参数:
    residuals: 残差序列
    save_path: 可选，保存结果的路径
    
    返回:
    best_order: 最佳ARIMA阶数
    model_info: 模型选择信息
    """
    print("开始寻找最佳ARIMA参数...")
    print(f"残差序列长度: {len(residuals)}")
    print(f"残差统计: 均值={np.mean(residuals):.4f}, 标准差={np.std(residuals):.4f}")
    
    # 调用核心函数寻找最佳参数
    best_order, best_model, model_info = select_best_arima_order(residuals)
    
    print(f"\n最佳ARIMA阶数: {best_order}")
    
    # 如果有保存路径，保存结果
    if save_path:
        # 保存ADF检验结果
        if model_info['adf_results']:
            adf_df = pd.DataFrame(model_info['adf_results'])
            adf_file = os.path.join(save_path, 'adf_results.csv')
            adf_df.to_csv(adf_file, index=False)
            print(f"ADF检验结果已保存到: {adf_file}")
        
        # 保存候选模型信息
        if model_info['candidate_models'] is not None:
            cand_df = model_info['candidate_models'].copy()
            cand_file = os.path.join(save_path, 'candidate_models.csv')
            cand_df.to_csv(cand_file, index=False)
            print(f"候选模型信息已保存到: {cand_file}")
            
            # 额外保存一个明确标记最佳模型的文件
            if len(cand_df) > 0:
                best_model_info = cand_df.iloc[0].copy()
                best_model_df = pd.DataFrame([best_model_info])
                best_model_file = os.path.join(save_path, '最佳ARIMA模型参数.csv')
                best_model_df.to_csv(best_model_file, index=False, encoding='utf-8-sig')
                print(f"最佳ARIMA模型参数已保存到: {best_model_file}")
        
        # 保存残差信息
        if 'residuals_info' in model_info:
            residuals_info = model_info['residuals_info']
            residuals_df = pd.DataFrame([residuals_info])
            residuals_file = os.path.join(save_path, 'residuals_info.csv')
            residuals_df.to_csv(residuals_file, index=False)
            print(f"残差信息已保存到: {residuals_file}")
    
    return best_order, best_model, model_info

# 主程序：在原文件夹中处理并保存结果
if __name__ == "__main__":
    # 设置路径
    data_folder = ''
    sdi_file = 'SDI.xlsx'
    
    # 读取数据函数
    def load_data(dementia_file, sdi_file):
        df_dementia = pd.read_excel(dementia_file)
        df_dementia = df_dementia[['Period', 'Prevalence']].sort_values('Period')
        
        df_sdi = pd.read_excel(sdi_file)
        df_sdi = df_sdi[['Period', 'SDI']].sort_values('Period')
        
        df = pd.merge(df_dementia, df_sdi, on='Period')
        df['Prevalence'] = df['Prevalence'] / 100000
        
        return df
    
    # 获取所有数据文件
    dementia_files = glob.glob(os.path.join(data_folder, '*.xlsx'))
    
    if not dementia_files:
        print(f"在文件夹 {data_folder} 中未找到Excel文件")
        exit()
    
    results = []
    
    for dementia_file in dementia_files:
        print(f"\n处理文件: {dementia_file}")
        file_name = os.path.basename(dementia_file)
        file_base_name = os.path.splitext(file_name)[0]
        
        try:
            # 加载数据
            df = load_data(dementia_file, sdi_file)
            print(f"成功读取数据: {len(df)}个数据点")
            
            # 准备数据
            prevalence = df['Prevalence'].values
            sdi_values = df['SDI'].values
            
            # 线性回归
            model_lr = LinearRegression()
            X = sdi_values.reshape(-1, 1)
            y = safe_log(prevalence)
            model_lr.fit(X, y)
            
            # 计算残差
            residuals = y - model_lr.predict(X)
            
            # 在原文件所在目录创建ARIMA参数文件夹
            file_dir = os.path.dirname(dementia_file)
            arima_results_folder = os.path.join(file_dir, "ARIMA参数结果")
            os.makedirs(arima_results_folder, exist_ok=True)
            
            # 为当前文件创建子文件夹
            file_results_folder = os.path.join(arima_results_folder, file_base_name)
            os.makedirs(file_results_folder, exist_ok=True)
            
            # 寻找最佳ARIMA参数
            best_order, best_model, model_info = find_best_arima_for_residuals(
                residuals, 
                save_path=file_results_folder
            )
            
            # 从排序后的候选模型中获取AIC最小的那个
            best_aic = None
            best_bic = None
            best_hqic = None
            
            if model_info['candidate_models'] is not None and len(model_info['candidate_models']) > 0:
                # 第一个就是AIC最小的
                best_row = model_info['candidate_models'].iloc[0]
                best_aic = best_row['aic']
                best_bic = best_row['bic']
                best_hqic = best_row.get('hqic')
            
            # 记录结果
            result = {
                'file': file_name,
                'best_order': best_order,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'best_hqic': best_hqic,
                'optimal_d': model_info['optimal_d'],
                'residuals_length': len(residuals),
                'results_folder': file_results_folder
            }
            results.append(result)
            
            # 在当前文件同目录下保存简洁结果
            simple_result = pd.DataFrame([{
                '文件': file_name,
                '最佳ARIMA阶数': str(best_order),
                'AIC': best_aic,
                'BIC': best_bic,
                'HQIC': best_hqic,
                '最优差分阶数d': model_info['optimal_d'],
                '残差序列长度': len(residuals),
                '选择标准': 'AIC最小',
                '详细结果文件夹': file_results_folder
            }])
            
            simple_result_file = os.path.join(file_dir, f"{file_base_name}_ARIMA参数.csv")
            simple_result.to_csv(simple_result_file, index=False, encoding='utf-8-sig')
            print(f"简洁参数结果已保存到: {simple_result_file}")
            
        except Exception as e:
            print(f"处理文件 {dementia_file} 时出错: {e}")
            continue
    
    # 在ARIMA参数结果文件夹中保存汇总结果
    if results:
        summary_df = pd.DataFrame(results)
        
        # 创建汇总文件夹（在原文件夹内）
        summary_folder = os.path.join(data_folder, "ARIMA参数结果")
        os.makedirs(summary_folder, exist_ok=True)
        
        summary_file = os.path.join(summary_folder, '所有文件_ARIMA参数汇总.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n所有文件ARIMA参数汇总已保存到: {summary_file}")
        
        # 打印汇总信息
        print("\n=== ARIMA参数选择汇总（按AIC最小选择） ===")
        for result in results:
            aic_str = f"{result['best_aic']:.2f}" if result['best_aic'] is not None else "N/A"
            print(f"{result['file']}: ARIMA{result['best_order']} (AIC={aic_str})")
    
    print(f"\n处理完成！所有结果已保存到原文件夹的'ARIMA参数结果'子文件夹中")