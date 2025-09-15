function generateVisualizationReport(results, save_dir, config)
%GENERATEVISUALIZATIONREPORT 生成可视化报告（修复版）
% 修复了数据缺失和极坐标图问题

if nargin < 3
    config = struct();
end

if nargin < 2 || isempty(save_dir)
    save_dir = 'results';
end

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

fprintf('📊 开始生成可视化报告...\n');

try
    % 数据验证和预处理
    validated_results = validateAndFillMissingData(results, config);
    
    % 生成各类图表
    fprintf('  - 攻击者策略图\n');
    generateAttackerStrategyChart_Fixed(validated_results, save_dir, config);
    
    fprintf('  - 防御者策略对比图\n');
    generateDefenderStrategiesChart_Fixed(validated_results, save_dir, config);
    
    fprintf('  - 性能指标分析图\n');
    generatePerformanceMetricsChart_Fixed(validated_results, save_dir, config);
    
    fprintf('  - 算法参数变化图\n');
    generateParameterChangesChart_Fixed(validated_results, save_dir, config);
    
    fprintf('  - 防御者性能对比图\n');
    generateDefenderComparisonChart_Fixed(validated_results, save_dir, config);
    
    % 生成HTML报告
    generateHTMLReport_Fixed(validated_results, save_dir, config);
    
    fprintf('✅ 可视化报告生成完成: 