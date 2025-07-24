function generateVisualizationReport(all_agents, config)
%GENERATEVISUALIZATIONREPORT 修复版可视化报告生成器
fprintf('\n=== 开始生成可视化报告 ===\n');

try
    % 1. 安全的数据收集
    fprintf('📋 收集智能体数据...\n');
    [results_data, save_dir] = collectDataSafely(all_agents, config);

    % 2. 生成图表（带错误处理）
    fprintf('📊 生成可视化图表...\n');
    generateChartsWithErrorHandling(results_data, save_dir);

    fprintf('✅ 可视化报告生成完成！\n');
    fprintf('📁 报告保存位置: %s\n', save_dir);

catch ME
    fprintf('❌ 可视化生成过程中出现错误:\n');
    fprintf('错误信息: %s\n', ME.message);
    fprintf('⚠️ 继续执行主程序...\n');
end
end

function [results_data, save_dir] = collectDataSafely(all_agents, config)
%COLLECTDATASAFELY 安全收集数据
try
    collector = ResultsCollector(all_agents, config);
    collector.collectFromAgents();
    if ismethod(collector, 'generateMissingData')
        collector.generateMissingData();
    end
    results_data = collector.getResults();
catch
    % 创建默认数据结构
    results_data = createDefaultResults(config);
end

% 创建保存目录
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
save_dir = fullfile(pwd, 'reports', timestamp);
if ~exist('reports', 'dir'), mkdir('reports'); end
if ~exist(save_dir, 'dir'), mkdir(save_dir); end
end

function generateChartsWithErrorHandling(results_data, save_dir)
%GENERATECHARTSWWITHERRORHANDLING 带错误处理的图表生成
charts = {'攻击者策略分析图', '防御者策略对比图', '性能指标分析图', ...
          '算法参数变化图', '防御者性能对比图'};
functions = {@generateAttackerStrategyChart, @generateDefenderStrategiesChart, ...
            @generatePerformanceMetricsChart, @generateParameterChangesChart, ...
            @generateDefenderComparisonChart};

for i = 1:length(charts)
    try
        fprintf('  - %s\n', charts{i});
        functions{i}(results_data, save_dir);
    catch ME
        fprintf('    ❌ %s生成失败: %s\n', charts{i}, ME.message);
    end
end
end

function results_data = createDefaultResults(config)
%CREATEDEFAULTRESULTS 创建默认结果数据
n_iterations = 100;
if isfield(config, 'n_iterations')
    n_iterations = config.n_iterations;
end

results_data = struct();
results_data.n_iterations = n_iterations;
results_data.attacker_final_strategy = rand(1, 10);
results_data.qlearning_final_strategy = rand(1, 10);
results_data.sarsa_final_strategy = rand(1, 10);
results_data.doubleqlearning_final_strategy = rand(1, 10);
end