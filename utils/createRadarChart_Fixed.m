function createRadarChart_Fixed(data, labels, metric_labels, chart_title)
%CREATERADARCHART_FIXED 创建修复后的雷达图
% 解决"不支持将极坐标图添加到 axes"的问题

if nargin < 4
    chart_title = '性能雷达图';
end

n_metrics = size(data, 2);
n_algorithms = size(data, 1);

if n_metrics ~= length(metric_labels)
    error('数据维度与标签数量不匹配');
end

% 数据预处理和归一化
data_clean = data;
data_clean(isnan(data_clean)) = 0;
data_clean(isinf(data_clean)) = 0;

