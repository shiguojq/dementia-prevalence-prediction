import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import os
import glob

# 1. 读取数据函数
def load_data(dementia_file, sdi_file):
    # 读取患病率数据
    df_dementia = pd.read_excel(dementia_file)
    df_dementia = df_dementia[['Period', 'Prevalence']].sort_values('Period')
    
    # 读取SDI数据
    df_sdi = pd.read_excel(sdi_file)
    df_sdi = df_sdi[['Period', 'SDI']].sort_values('Period')
    
    # 合并数据
    df = pd.merge(df_dementia, df_sdi, on='Period')
    
    # 转换患病率为比例
    df['Prevalence'] = df['Prevalence'] / 100000
    
    return df

# 2. 安全log转换函数
def safe_log(x):
    x = np.where(x <= 0, 0.0001, x)
    x = np.where(x >= 1, 0.9999, x)
    return np.log(x)

# 3. 预测函数 - 修改为同时返回历史拟合值
def doprediction_with_sdi_arima(df, future_years, future_sdi):
    # 准备数据
    years = df['Period'].values
    prevalence = df['Prevalence'].values
    sdi_values = df['SDI'].values
    
    # 转换为Series用于预测
    ts_data = pd.Series(prevalence, index=years)
    
    # 线性回归部分
    model_lr = LinearRegression()
    X = sdi_values.reshape(-1, 1)
    y = safe_log(prevalence)
    model_lr.fit(X, y)
    
    # 预测未来log值
    future_X = future_sdi.reshape(-1, 1)
    future_log_pred = model_lr.predict(future_X)
    
    # 计算历史拟合值和残差
    historical_log_fitted = model_lr.predict(X)
    residuals = y - historical_log_fitted
    
    # 转换历史拟合值为原始尺度
    historical_fitted = np.exp(historical_log_fitted)
    historical_fitted = np.clip(historical_fitted, 0, 1)
    
    # ARIMA部分
    model_arima = ARIMA(residuals, order=(0, 1, 1))
    model_arima_fit = model_arima.fit()
    residual_forecast = model_arima_fit.forecast(steps=len(future_years))
    
    # 综合预测
    future_log_pred_with_residual = future_log_pred + residual_forecast
    future_pred = np.exp(future_log_pred_with_residual)
    future_pred = np.clip(future_pred, 0, 1)
    
    return future_pred, historical_fitted

# 4. 自助法不确定性估计 - 修改为同时返回历史拟合值
def bootstrap_prediction(df, future_years, future_sdi, n_iterations=100):
    predictions = []
    historical_fits = []
    years = df['Period'].values
    prevalence = df['Prevalence'].values
    
    for _ in range(n_iterations):
        # 添加噪声
        noise = np.random.normal(0, prevalence.std()/3, size=len(prevalence))
        perturbed_prevalence = np.clip(prevalence + noise, 0.001, 0.999)
        
        # 创建扰动后的DataFrame
        perturbed_df = df.copy()
        perturbed_df['Prevalence'] = perturbed_prevalence
        
        try:
            pred, hist_fit = doprediction_with_sdi_arima(perturbed_df, future_years, future_sdi)
            predictions.append(pred)
            historical_fits.append(hist_fit)
        except:
            continue
    
    if not predictions:
        return None, None, None, None, None, None
    
    predictions = np.array(predictions)
    historical_fits = np.array(historical_fits)
    
    # 计算未来预测的置信区间
    future_lower = np.percentile(predictions, 2.5, axis=0)
    future_upper = np.percentile(predictions, 97.5, axis=0)
    future_median = np.percentile(predictions, 50, axis=0)
    
    # 计算历史拟合值的置信区间
    hist_lower = np.percentile(historical_fits, 2.5, axis=0)
    hist_upper = np.percentile(historical_fits, 97.5, axis=0)
    hist_median = np.percentile(historical_fits, 50, axis=0)
    
    return future_lower, future_upper, future_median, hist_lower, hist_upper, hist_median

# 5. 可视化函数 - 修改为支持多组数据
def plot_results(all_results, output_folder):
    plt.figure(figsize=(12, 6))
    
    # 为每组数据绘制图表
    colors = plt.cm.tab10.colors  # 使用不同的颜色
    for i, (df, future_years, point_pred, future_lower, future_upper, future_median, hist_median, hist_lower, hist_upper, file_name) in enumerate(all_results):
        color = colors[i % len(colors)]
        label = os.path.splitext(os.path.basename(file_name))[0]
        
        # 历史数据
        plt.plot(df['Period'], df['Prevalence']*100000, 'o-', color=color, label=f'{label} - Historical', linewidth=2)
        
        # 历史拟合值
        plt.plot(df['Period'], hist_median*100000, ':', color=color, linewidth=1.5, label=f'{label} - Fitted Trend')
        plt.fill_between(df['Period'], hist_lower*100000, hist_upper*100000, 
                        color=color, alpha=0.1, label=f'{label} - Historical CI')
        
        # 预测数据
        plt.plot(future_years, point_pred*100000, '--', color=color, linewidth=2, label=f'{label} - Predicted')
        plt.fill_between(future_years, future_lower*100000, future_upper*100000, 
                        color=color, alpha=0.2, label=f'{label} - Future CI')
    
    # 图表装饰
    plt.title('Dementia Prevalence Projection with Historical Fitting (Multiple Groups)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Prevalence (per 100,000)', fontsize=12)
    max_historical_year = max(max(df['Period']) for df, *_ in all_results)
    plt.axvline(x=max_historical_year, color='gray', linestyle=':', label='Prediction Start')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_folder, 'combined_dementia_prevalence_projection_with_fitting.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

# 主程序
if __name__ == "__main__":
    # 文件路径设置
    dementia_folder = ''
    sdi_file = 'SDI.xlsx'  # SDI文件位置保持不变
    output_folder = dementia_folder
    
    # 获取所有dementia Excel文件
    dementia_files = glob.glob(os.path.join(dementia_folder, '*.xlsx'))
    
    if not dementia_files:
        print(f"在文件夹 {dementia_folder} 中未找到Excel文件")
        exit()
    
    all_results = []
    all_pred_results = []
    all_hist_results = []
    
    for dementia_file in dementia_files:
        print(f"\n处理文件: {dementia_file}")
        
        # 加载数据
        try:
            df = load_data(dementia_file, sdi_file)
            print(f"成功读取数据: {len(df)}个数据点")
        except Exception as e:
            print(f"读取文件错误: {e}")
            continue
        
        # 定义预测年份范围
        future_years = np.arange(max(df['Period'])+1, 2051)  # 预测到2050年
        
        # 获取未来SDI值
        df_sdi_future = pd.read_excel(sdi_file)
        future_sdi = df_sdi_future[df_sdi_future['Period'].isin(future_years)]['SDI'].values
        
        # 进行预测
        try:
            point_pred, hist_fitted = doprediction_with_sdi_arima(df, future_years, future_sdi)
            future_lower, future_upper, future_median, hist_lower, hist_upper, hist_median = bootstrap_prediction(df, future_years, future_sdi)
            print("预测和历史拟合完成")
        except Exception as e:
            print(f"预测错误: {e}")
            continue
        
        # 保存结果用于合并图表
        all_results.append((df, future_years, point_pred, future_lower, future_upper, future_median, hist_median, hist_lower, hist_upper, dementia_file))
        
        # 保存未来预测结果
        file_name = os.path.splitext(os.path.basename(dementia_file))[0]
        pred_results = pd.DataFrame({
            'Year': future_years,
            'Predicted_Prevalence': point_pred*100000,
            'Lower_Bound': future_lower*100000,
            'Upper_Bound': future_upper*100000,
            'Median_Prediction': future_median*100000,
            'SDI': future_sdi,
            'Group': file_name,
            'Data_Type': 'Future_Prediction'
        })
        all_pred_results.append(pred_results)
        
        # 保存历史拟合结果
        hist_results = pd.DataFrame({
            'Year': df['Period'].values,
            'Historical_Prevalence': df['Prevalence'].values*100000,
            'Fitted_Trend': hist_median*100000,
            'Fitted_Lower_Bound': hist_lower*100000,
            'Fitted_Upper_Bound': hist_upper*100000,
            'SDI': df['SDI'].values,
            'Group': file_name,
            'Data_Type': 'Historical_Fitting'
        })
        all_hist_results.append(hist_results)
        
        # 保存单个文件的完整结果到Excel
        output_file = os.path.join(output_folder, f"{file_name}_full_results.xlsx")
        with pd.ExcelWriter(output_file) as writer:
            # 写入历史拟合结果
            hist_results.to_excel(writer, sheet_name='Historical_Fitting', index=False)
            # 写入未来预测结果
            pred_results.to_excel(writer, sheet_name='Future_Prediction', index=False)
        print(f"完整结果已保存到: {output_file}")
    
    if not all_results:
        print("没有成功处理任何文件")
        exit()
    
    # 合并所有预测结果到一个DataFrame
    combined_pred_results = pd.concat(all_pred_results, ignore_index=True)
    combined_hist_results = pd.concat(all_hist_results, ignore_index=True)
    
    # 可视化并保存结果
    plot_path = plot_results(all_results, output_folder)
    
    # 保存合并的预测结果为CSV
    output_csv_pred = os.path.join(output_folder, 'combined_dementia_prevalence_predictions.csv')
    combined_pred_results.to_csv(output_csv_pred, index=False)
    
    # 保存合并的历史拟合结果为CSV
    output_csv_hist = os.path.join(output_folder, 'combined_dementia_historical_fitting.csv')
    combined_hist_results.to_csv(output_csv_hist, index=False)
    
    # 保存合并的完整结果到Excel
    output_excel = os.path.join(output_folder, 'combined_dementia_full_results.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        combined_hist_results.to_excel(writer, sheet_name='Historical_Fitting', index=False)
        combined_pred_results.to_excel(writer, sheet_name='Future_Prediction', index=False)
    
    print(f"\n所有预测结果已合并保存到: {output_csv_pred}")
    print(f"所有历史拟合结果已合并保存到: {output_csv_hist}")
    print(f"完整合并结果已保存到Excel: {output_excel}")
    print(f"合并可视化图表已保存到: {plot_path}")