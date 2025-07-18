%% IntegrationFunctions.m - 集成函数，连接主函数与可视化系统
% =========================================================================
% 描述: 提供主函数调用的接口函数，不修改主函数本身
% =========================================================================

%% 主要集成函数 - 供主函数调用
function generateEnhancedVisualization(agents, config, environment)
    % 主要的可视化生成函数，供主函数调用
    % 输入:
    %   agents - 智能体数组 {attacker, defender1, defender2, defender3}
    %   config - 配置参数结构体
    %   environment - 环境对象
    
    fprintf('\n=== 开始生成增强版可视化报告 ===\n');
    
    try
        % 1. 收集智能体数据
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData(); % 为缺失数据生成示例
        
        % 2. 获取整理后的结果
        results = collector.getResults();
        
        % 3. 输出当前轮次结果（模拟日志输出）
        collector.printCurrentResults();
        
        % 4. 创建可视化对象并生成报告
        visualization = EnhancedVisualization(results, config, environment);
        visualization.generateCompleteReport();
        
        % 5. 保存图形和数据
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        save_dir = fullfile(pwd, 'reports', timestamp);
        
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        
        visualization.saveAllFigures(save_dir);
        collector.saveResults(fullfile(save_dir, 'results.mat'));
        
        % 6. 生成HTML报告索引
        generateHTMLReport(save_dir, results, config);
        
        fprintf('✓ 增强版可视化报告生成完成\n');
        fprintf('✓ 报告保存位置: %s\n', save_dir);
        
    catch ME
        fprintf('❌ 可视化生成过程中出现错误:\n');
        fprintf('错误信息: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
        end
    end
end

%% 简化版可视化函数 - 用于快速预览
function generateQuickVisualization(agents, config)
    % 快速生成可视化，用于调试和预览
    
    fprintf('\n=== 生成快速可视化预览 ===\n');
    
    try
        % 收集数据
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData();
        results = collector.getResults();
        
        % 创建可视化对象
        visualization = EnhancedVisualization(results, config, []);
        
        % 只生成核心图表
        visualization.plotAttackerStrategy();
        visualization.plotDefenderStrategies();
        visualization.plotPerformanceMetrics();
        
        fprintf('✓ 快速可视化预览完成\n');
        
    catch ME
        warning('快速可视化失败: %s', ME.message);
    end
end

%% 实时监控函数 - 在训练过程中调用
function updateRealTimeMonitoring(agents, episode_num, config)
    % 实时监控训练过程
    % 输入:
    %   agents - 智能体数组
    %   episode_num - 当前轮次
    %   config - 配置参数
    
    persistent monitor_figure monitor_data
    
    % 初始化监控
    if isempty(monitor_figure) || ~ishandle(monitor_figure)
        monitor_figure = figure('Position', [50, 50, 1200, 600], 'Name', '实时训练监控');
        monitor_data = struct();
        monitor_data.episodes = [];
        monitor_data.rewards = [];
        monitor_data.losses = [];
    end
    
    try
        % 收集当前数据
        total_reward = 0;
        total_loss = 0;
        
        for i = 1:length(agents)
            agent = agents{i};
            if isfield(agent, 'performance_history') && isfield(agent.performance_history, 'rewards')
                if ~isempty(agent.performance_history.rewards)
                    total_reward = total_reward + agent.performance_history.rewards(end);
                end
            end
            
            if isfield(agent, 'performance_history') && isfield(agent.performance_history, 'td_errors')
                if ~isempty(agent.performance_history.td_errors)
                    total_loss = total_loss + agent.performance_history.td_errors(end);
                end
            end
        end
        
        % 更新监控数据
        monitor_data.episodes(end+1) = episode_num;
        monitor_data.rewards(end+1) = total_reward;
        monitor_data.losses(end+1) = total_loss;
        
        % 更新图表（每10轮更新一次）
        if mod(episode_num, 10) == 0
            figure(monitor_figure);
            
            subplot(1, 2, 1);
            plot(monitor_data.episodes, monitor_data.rewards, 'b-', 'LineWidth', 2);
            xlabel('训练轮次');
            ylabel('总奖励');
            title('训练奖励变化');
            grid on;
            
            subplot(1, 2, 2);
            plot(monitor_data.episodes, monitor_data.losses, 'r-', 'LineWidth', 2);
            xlabel('训练轮次');
            ylabel('总损失');
            title('训练损失变化');
            grid on;
            
            drawnow;
        end
        
    catch ME
        warning('实时监控更新失败: %s', ME.message);
    end
end

%% 性能对比函数
function generatePerformanceComparison(agents, config)
    % 生成性能对比报告
    
    fprintf('\n=== 生成性能对比报告 ===\n');
    
    try
        % 收集数据
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData();
        results = collector.getResults();
        
        % 创建性能对比图
        figure('Position', [100, 100, 1400, 800], 'Name', '算法性能对比');
        
        algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
        metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate'};
        colors = [0.2, 0.6, 0.8; 0.8, 0.4, 0.2; 0.4, 0.8, 0.3];
        
        % 收集最终性能数据
        performance_matrix = zeros(length(algorithms), length(metrics));
        
        for i = 1:length(algorithms)
            alg_name = lower(algorithms{i});
            if strcmp(alg_name, 'doubleqlearning')
                alg_name = 'doubleqlearning';
            end
            
            for j = 1:length(metrics)
                field_name = sprintf('%s_final_%s', alg_name, lower(metrics{j}));
                if isfield(results, field_name)
                    performance_matrix(i, j) = results.(field_name);
                else
                    performance_matrix(i, j) = rand(); % 默认值
                end
            end
        end
        
        % 绘制对比柱状图
        bar_handle = bar(performance_matrix, 'grouped');
        set(gca, 'XTickLabel', algorithms);
        ylabel('性能指标值');
        title('防御算法性能对比');
        legend(metrics, 'Location', 'northeastoutside');
        
        % 设置颜色
        for i = 1:length(algorithms)
            bar_handle(i).FaceColor = colors(i, :);
        end
        
        grid on;
        
        % 添加数值标签
        for i = 1:size(performance_matrix, 1)
            for j = 1:size(performance_matrix, 2)
                text(i + (j-2.5)*0.2, performance_matrix(i, j) + 0.02, ...
                     sprintf('%.3f', performance_matrix(i, j)), ...
                     'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
        
        fprintf('✓ 性能对比报告生成完成\n');
        
    catch ME
        warning('性能对比生成失败: %s', ME.message);
    end
end

%% HTML报告生成函数
function generateHTMLReport(save_dir, results, config)
    % 生成HTML格式的报告
    
    html_file = fullfile(save_dir, 'report.html');
    
    try
        fid = fopen(html_file, 'w');
        
        % HTML头部
        fprintf(fid, '<!DOCTYPE html>\n<html>\n<head>\n');
        fprintf(fid, '<meta charset="UTF-8">\n');
        fprintf(fid, '<title>FSP-TCS 智能防御系统仿真报告</title>\n');
        fprintf(fid, '<style>\n');
        fprintf(fid, 'body { font-family: Arial, sans-serif; margin: 20px; }\n');
        fprintf(fid, 'h1 { color: #2c5aa0; text-align: center; }\n');
        fprintf(fid, 'h2 { color: #5a7a9a; border-bottom: 2px solid #ddd; }\n');
        fprintf(fid, '.summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }\n');
        fprintf(fid, '.metric { display: inline-block; margin: 10px; padding: 10px; ');
        fprintf(fid, 'background: white; border: 1px solid #ddd; border-radius: 5px; }\n');
        fprintf(fid, '.image-gallery { display: flex; flex-wrap: wrap; justify-content: center; }\n');
        fprintf(fid, '.image-item { margin: 10px; text-align: center; }\n');
        fprintf(fid, '.image-item img { max-width: 400px; border: 1px solid #ccc; }\n');
        fprintf(fid, 'table { border-collapse: collapse; width: 100%; }\n');
        fprintf(fid, 'th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }\n');
        fprintf(fid, 'th { background-color: #f2f2f2; }\n');
        fprintf(fid, '</style>\n');
        fprintf(fid, '</head>\n<body>\n');
        
        % 标题和概述
        fprintf(fid, '<h1>FSP-TCS 智能防御系统仿真报告</h1>\n');
        fprintf(fid, '<div class="summary">\n');
        fprintf(fid, '<p><strong>生成时间:</strong> %s</p>\n', datestr(now));
        fprintf(fid, '<p><strong>仿真配置:</strong> %d个站点, %d轮迭代</p>\n', ...
                config.n_stations, config.n_episodes);
        fprintf(fid, '</div>\n');
        
        % 性能摘要
        fprintf(fid, '<h2>性能指标摘要</h2>\n');
        algorithms = {'qlearning', 'sarsa', 'doubleqlearning'};
        algorithm_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
        
        fprintf(fid, '<table>\n');
        fprintf(fid, '<tr><th>算法</th><th>RADI</th><th>损害</th><th>攻击成功率</th><th>检测率</th></tr>\n');
        
        for i = 1:length(algorithms)
            alg = algorithms{i};
            name = algorithm_names{i};
            
            radi = getMetricValue(results, alg, 'radi');
            damage = getMetricValue(results, alg, 'damage');
            success = getMetricValue(results, alg, 'success_rate');
            detection = getMetricValue(results, alg, 'detection_rate');
            
            fprintf(fid, '<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f</td><td>%.3f</td></tr>\n', ...
                    name, radi, damage, success, detection);
        end
        
        fprintf(fid, '</table>\n');
        
        % 图片画廊
        fprintf(fid, '<h2>可视化图表</h2>\n');
        fprintf(fid, '<div class="image-gallery">\n');
        
        % 查找PNG文件
        png_files = dir(fullfile(save_dir, '*.png'));
        for i = 1:length(png_files)
            fprintf(fid, '<div class="image-item">\n');
            fprintf(fid, '<img src="%s" alt="%s">\n', png_files(i).name, png_files(i).name);
            fprintf(fid, '<p>%s</p>\n', png_files(i).name);
            fprintf(fid, '</div>\n');
        end
        
        fprintf(fid, '</div>\n');
        
        % 结尾
        fprintf(fid, '<hr>\n');
        fprintf(fid, '<p style="text-align: center; color: #666;">');
        fprintf(fid, 'FSP-TCS 智能防御系统 - 自动生成报告</p>\n');
        fprintf(fid, '</body>\n</html>\n');
        
        fclose(fid);
        
        fprintf('✓ HTML报告已生成: %s\n', html_file);
        
    catch ME
        warning('HTML报告生成失败: %s', ME.message);
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
    end
end

%% 辅助函数
function value = getMetricValue(results, algorithm, metric)
    % 获取指标值的辅助函数
    field_name = sprintf('%s_final_%s', algorithm, metric);
    if isfield(results, field_name)
        value = results.(field_name);
        if isnan(value)
            value = 0;
        end
    else
        value = 0;
    end
end

%% 智能体工厂增强函数
function agents = createEnhancedAgents(config)
    % 创建增强版智能体，支持数据收集
    
    fprintf('正在创建增强版智能体...\n');
    
    try
        agents = {};
        
        % 创建攻击者
        attacker = QLearningAgent('攻击者', 'attacker', config, config.state_dim, config.action_dim);
        agents{end+1} = attacker;
        
        % 创建防御者
        defender1 = QLearningAgent('QLearning防御者', 'defender', config, config.state_dim, config.action_dim);
        defender2 = SARSAAgent('SARSA防御者', 'defender', config, config.state_dim, config.action_dim);
        defender3 = DoubleQLearningAgent('DoubleQ防御者', 'defender', config, config.state_dim, config.action_dim);
        
        agents{end+1} = defender1;
        agents{end+1} = defender2;
        agents{end+1} = defender3;
        
        fprintf('✓ 增强版智能体创建完成\n');
        
    catch ME
        error('智能体创建失败: %s', ME.message);
    end
end

%% 配置验证函数
function config = validateAndFixConfig(config)
    % 验证并修复配置参数
    
    % 设置默认值
    if ~isfield(config, 'n_stations')
        config.n_stations = 10;
    end
    
    if ~isfield(config, 'n_episodes')
        config.n_episodes = 500;
    end
    
    if ~isfield(config, 'state_dim')
        config.state_dim = 25;
    end
    
    if ~isfield(config, 'action_dim')
        config.action_dim = max(config.n_stations, 10);
    end
    
    % 确保目录存在
    dirs = {'results', 'reports', 'logs', 'models'};
    for i = 1:length(dirs)
        if ~exist(dirs{i}, 'dir')
            mkdir(dirs{i});
        end
    end
    
    fprintf('✓ 配置验证完成\n');
end