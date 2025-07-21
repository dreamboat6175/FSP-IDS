%% generateEnhancedVisualization.m - 集成与可视化调度中心 (v3.0)
% =========================================================================
% 描述:
%   此文件作为主程序与底层仿真、数据收集和可视化模块之间的桥梁。
%   它提供了一系列高级接口函数，用于启动可视化流程、生成报告、
%   进行实时监控和性能对比。此版本根据您的完整项目结构进行了重构，
%   动态地从`agents`对象中提取信息，并与您项目中的其他模块（如
%   ResultsCollector, EnhancedVisualization）无缝协作。
%
% 主要功能:
%   - generateEnhancedVisualization: 生成包含所有图表和HTML的完整报告。
%   - generateQuickVisualization: 生成核心图表的快速预览，用于调试。
%   - updateRealTimeMonitoring: 在训练循环中被调用，以实时更新监控图表。
%   - generatePerformanceComparison: 动态生成防御算法的性能对比图。
%   - generateHTMLReport: 动态创建一个包含仿真结果摘要和图表的HTML报告。
%
% 使用说明:
%   在您的主脚本 (main_fsp.m) 的末尾调用此文件中的函数，例如:
%   >> generateEnhancedVisualization(agents, config, environment);
%
% =========================================================================

%% 主集成函数 - 生成完整可视化报告
function generateEnhancedVisualization(agents, config, environment)
    % 生成完整的增强版可视化报告，包括所有图表、数据和HTML索引。
    % 这是推荐在仿真结束后调用的主要函数。
    
    fprintf('\n\n==================================================\n');
    fprintf('=== 开始生成增强版可视化报告 ===\n');
    fprintf('==================================================\n');
    
    try
        % 步骤 1: 验证和标准化配置
        % 确保 n_episodes 存在，这是所有历史数据长度的基准
        config = validateAndFixConfig(config);

        % 步骤 2: 收集和整理数据
        fprintf('1. 正在从智能体收集数据...\n');
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData(); % 关键: 此处会使用 config.n_episodes 生成示例数据
        results = collector.getResults();
        fprintf('   ✓ 数据收集完成。\n');
        
        % 步骤 3: 输出当前轮次的摘要结果 (模拟日志)
        collector.printCurrentResults();
        
        % 步骤 4: 创建可视化对象并生成所有图表
        fprintf('2. 正在生成所有可视化图表...\n');
        % 假设 visualization/EnhancedVisualization.m 在路径中
        visualization = EnhancedVisualization(results, config, environment);
        visualization.generateCompleteReport(); % 此方法会调用所有绘图函数
        fprintf('   ✓ 图表生成完成。\n');
        
        % 步骤 5: 保存所有生成的图表和结果数据
        fprintf('3. 正在保存报告文件...\n');
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        save_dir = fullfile(pwd, 'reports', timestamp);
        
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        
        visualization.saveAllFigures(save_dir);
        collector.saveResults(fullfile(save_dir, 'results.mat'));
        fprintf('   ✓ 图表和数据已保存。\n');
        
        % 步骤 6: 生成HTML报告作为导航索引
        fprintf('4. 正在生成HTML索引报告...\n');
        generateHTMLReport(save_dir, results, config, agents); % 传递 agents 以动态获取名称
        fprintf('   ✓ HTML报告已生成。\n');
        
        fprintf('\n✓ 增强版可视化报告生成成功!\n');
        fprintf('✓ 报告保存位置: %s\n', save_dir);
        
    catch ME
        fprintf(2, '\n❌ 可视化生成过程中出现严重错误:\n');
        fprintf(2, '错误类型: %s\n', ME.identifier);
        fprintf(2, '错误信息: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf(2, '出错文件: %s\n', ME.stack(1).file);
            fprintf(2, '出错函数: %s, 行号: %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf(2, '请检查相关代码和输入数据。\n');
    end
end

%% 快速预览函数
function generateQuickVisualization(agents, config)
    % 生成一个快速的可视化预览，仅包含核心图表，用于调试。
    
    fprintf('\n=== 生成快速可视化预览 ===\n');
    
    try
        config = validateAndFixConfig(config);
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData();
        results = collector.getResults();
        
        visualization = EnhancedVisualization(results, config, []);
        
        % 只生成几个核心图表用于快速检查
        visualization.plotAttackerStrategy();
        visualization.plotDefenderStrategies();
        visualization.plotPerformanceMetrics();
        
        fprintf('✓ 快速可视化预览完成。\n');
        
    catch ME
        warning('快速可视化失败: %s', ME.message);
    end
end

%% 实时监控函数
function updateRealTimeMonitoring(agents, episode_num)
    % 在训练过程中实时更新奖励和损失曲线，用于监控训练状态。
    % 使用 persistent 变量来存储历史数据。
    
    persistent monitor_figure monitor_data;
    
    % 首次调用时初始化图形和数据结构
    if isempty(monitor_figure) || ~ishandle(monitor_figure)
        monitor_figure = figure('Position', [50, 50, 1200, 500], 'Name', '实时训练监控', 'NumberTitle', 'off');
        monitor_data = struct('episodes', [], 'rewards', [], 'losses', []);
        fprintf('初始化实时监控窗口...\n');
    end
    
    try
        % 收集当前轮次的数据
        current_reward = 0;
        current_loss = 0;
        
        for i = 1:length(agents)
            agent = agents{i};
            % 从 agent 的 performance_history 中获取最新数据
            if isprop(agent, 'performance_history') && isstruct(agent.performance_history)
                if isfield(agent.performance_history, 'rewards') && ~isempty(agent.performance_history.rewards)
                    current_reward = current_reward + agent.performance_history.rewards(end);
                end
                if isfield(agent.performance_history, 'td_errors') && ~isempty(agent.performance_history.td_errors)
                    current_loss = current_loss + agent.performance_history.td_errors(end);
                end
            end
        end
        
        % 更新监控数据
        monitor_data.episodes(end+1) = episode_num;
        monitor_data.rewards(end+1) = current_reward;
        monitor_data.losses(end+1) = current_loss;
        
        % 为避免性能问题，设置更新频率 (例如每10轮更新一次)
        update_frequency = 10;
        if mod(episode_num, update_frequency) == 0 || episode_num == 1
            figure(monitor_figure); % 激活监控窗口
            
            % 绘制奖励图
            subplot(1, 2, 1);
            plot(monitor_data.episodes, monitor_data.rewards, 'b-', 'LineWidth', 2);
            xlabel('训练轮次');
            ylabel('累积奖励');
            title(sprintf('训练奖励变化 (更新于第 %d 轮)', episode_num));
            grid on;
            
            % 绘制损失图
            subplot(1, 2, 2);
            plot(monitor_data.episodes, monitor_data.losses, 'r-', 'LineWidth', 2);
            xlabel('训练轮次');
            ylabel('累积损失 (TD-Error)');
            title(sprintf('训练损失变化 (更新于第 %d 轮)', episode_num));
            grid on;
            
            drawnow; % 强制刷新图形
        end
        
    catch ME
        warning('实时监控更新失败: %s', ME.message);
    end
end

%% 性能对比函数
function generatePerformanceComparison(agents, config)
    % 动态生成不同防御算法在关键性能指标上的对比柱状图。
    
    fprintf('\n=== 生成算法性能对比报告 ===\n');
    
    try
        config = validateAndFixConfig(config);
        collector = ResultsCollector(agents, config);
        collector.collectFromAgents();
        collector.generateMissingData();
        results = collector.getResults();
        
        figure('Position', [100, 100, 1200, 700], 'Name', '算法性能对比', 'NumberTitle', 'off');
        
        % 动态获取防御者信息
        defenders = getDefenderInfo(agents);
        if isempty(defenders)
            warning('未找到任何防御者智能体，无法生成性能对比图。');
            close(gcf);
            return;
        end
        
        algorithms = {defenders.displayName};
        metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate'};
        colors = lines(length(algorithms)); 
        
        % 收集最终性能数据
        performance_matrix = zeros(length(algorithms), length(metrics));
        
        for i = 1:length(algorithms)
            alg_key = defenders(i).key;
            for j = 1:length(metrics)
                metric_key = lower(metrics{j});
                performance_matrix(i, j) = getMetricValue(results, alg_key, metric_key);
            end
        end
        
        % 绘制分组柱状图
        bar_handle = bar(performance_matrix, 'grouped');
        
        % 设置图表属性
        set(gca, 'XTickLabel', algorithms, 'FontSize', 10);
        ylabel('性能指标值 (越低越好，除检测率外)', 'FontSize', 12);
        title('防御算法最终性能对比', 'FontSize', 16);
        legend(metrics, 'Location', 'northeastoutside', 'FontSize', 10);
        grid on;
        ylim([0, max(performance_matrix(:), [], 'all') * 1.2 + 0.05]); % 动态调整Y轴范围
        
        % 为每个柱子设置颜色
        for i = 1:length(bar_handle)
            bar_handle(i).FaceColor = colors(i, :);
        end
        
        % 在柱状图上添加精确数值标签
        for i = 1:length(bar_handle)
            for j = 1:length(bar_handle(i).XData)
                x_pos = bar_handle(i).XData(j) + bar_handle(i).XOffset;
                y_pos = bar_handle(i).YData(j);
                text(x_pos, y_pos, sprintf('%.3f', y_pos), ...
                     'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'bottom', ...
                     'FontSize', 8, 'Rotation', 90);
            end
        end
        
        fprintf('✓ 性能对比报告生成完成。\n');
        
    catch ME
        warning('性能对比报告生成失败: %s', ME.message);
    end
end

%% HTML报告生成函数
function generateHTMLReport(save_dir, results, config, agents)
    % 生成一个HTML文件，作为所有结果的摘要和导航。
    
    html_file = fullfile(save_dir, 'report.html');
    
    try
        fid = fopen(html_file, 'w', 'n', 'UTF-8'); % 确保UTF-8编码
        
        % HTML 头部和CSS样式
        fprintf(fid, '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n');
        fprintf(fid, '<meta charset="UTF-8">\n');
        fprintf(fid, '<title>FSP-TCS 智能防御系统仿真报告</title>\n');
        fprintf(fid, '<style>%s</style>\n', getHTMLStyle());
        fprintf(fid, '</head>\n<body>\n');
        
        % 报告主标题
        fprintf(fid, '<h1>FSP-TCS 智能防御系统仿真报告</h1>\n');
        
        % 仿真概述
        fprintf(fid, '<div class="summary">\n');
        fprintf(fid, '<h2>仿真概述</h2>\n');
        fprintf(fid, '<ul>\n');
        fprintf(fid, '<li><strong>报告生成时间:</strong> %s</li>\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
        fprintf(fid, '<li><strong>仿真轮次:</strong> %d</li>\n', config.n_episodes);
        fprintf(fid, '<li><strong>网络站点数:</strong> %d</li>\n', config.n_stations);
        fprintf(fid, '<li><strong>状态空间维度:</strong> %d</li>\n', config.state_dim);
        fprintf(fid, '<li><strong>动作空间维度:</strong> %d</li>\n', config.action_dim);
        fprintf(fid, '</ul>\n</div>\n');
        
        % 性能指标表格
        fprintf(fid, '<h2>最终性能指标对比</h2>\n');
        fprintf(fid, '%s\n', generatePerformanceTableHTML(results, agents)); % 传递 agents
        
        % 可视化图表画廊
        fprintf(fid, '<h2>可视化图表</h2>\n');
        fprintf(fid, '<div class="image-gallery">\n');
        png_files = dir(fullfile(save_dir, '*.png'));
        if isempty(png_files)
            fprintf(fid, '<p>未找到任何图表文件 (*.png)。</p>\n');
        else
            for i = 1:length(png_files)
                img_name = png_files(i).name;
                img_title = strrep(img_name, '_', ' ');
                img_title = strrep(img_title, '.png', '');
                fprintf(fid, '<div class="image-item">\n');
                fprintf(fid, '<h3>%s</h3>\n', img_title);
                fprintf(fid, '<a href="%s" target="_blank"><img src="%s" alt="%s"></a>\n', img_name, img_name, img_title);
                fprintf(fid, '</div>\n');
            end
        end
        fprintf(fid, '</div>\n');
        
        % HTML 结尾
        fprintf(fid, '<footer>FSP-TCS 智能防御系统 - 自动生成报告</footer>\n');
        fprintf(fid, '</body>\n</html>\n');
        
        fclose(fid);
        
        fprintf('✓ HTML报告已成功生成: %s\n', html_file);
        
    catch ME
        warning('HTML报告生成失败: %s', ME.message);
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
    end
end

%% ------------------- 辅助函数 -------------------

%% HTML辅助函数 - 生成性能表格
function html_table = generatePerformanceTableHTML(results, agents)
    % 动态地从results结构体和agents列表生成HTML性能表格
    defenders = getDefenderInfo(agents);
    metrics = {'radi', 'damage', 'success_rate', 'detection_rate'};
    metric_names = {'RADI', '损害度', '攻击成功率', '检测率'};
    
    html_table = '<table>\n<tr><th>算法</th>';
    for i = 1:length(metric_names)
        html_table = [html_table, sprintf('<th>%s</th>', metric_names{i})];
    end
    html_table = [html_table, '</tr>\n'];
    
    if isempty(defenders)
        html_table = [html_table, '<tr><td colspan="%d">未找到防御者数据</td></tr>\n'];
    else
        for i = 1:length(defenders)
            defender = defenders(i);
            html_table = [html_table, sprintf('<tr><td><strong>%s</strong></td>', defender.displayName)];
            for j = 1:length(metrics)
                metric = metrics{j};
                value = getMetricValue(results, defender.key, metric);
                html_table = [html_table, sprintf('<td>%.4f</td>', value)];
            end
            html_table = [html_table, '</tr>\n'];
        end
    end
    html_table = [html_table, '</table>\n'];
end

%% HTML辅助函数 - 定义CSS样式
function style = getHTMLStyle()
    % 返回HTML报告的CSS样式字符串，使主函数更整洁。
    style = [ ...
        'body { font-family: "Segoe UI", Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }', ...
        'h1, h2, h3 { color: #2c3e50; }', ...
        'h1 { text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }', ...
        'h2 { border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-top: 40px; }', ...
        '.summary { background: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; }', ...
        '.summary ul { list-style-type: none; padding-left: 0; }', ...
        '.image-gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }', ...
        '.image-item { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.2s; }', ...
        '.image-item:hover { transform: scale(1.03); }', ...
        '.image-item img { max-width: 450px; height: auto; border-radius: 5px; cursor: pointer; }', ...
        'table { border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }', ...
        'th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }', ...
        'th { background-color: #3498db; color: white; }', ...
        'tr:nth-child(even) { background-color: #f2f2f2; }', ...
        'footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #777; }' ...
    ];
end

%% 辅助函数 - 安全获取指标值
function value = getMetricValue(results, algorithm_key, metric)
    % 从 results 结构体中安全地获取指标值。
    % 如果字段不存在或值为NaN/空，则返回0。
    field_name = sprintf('%s_final_%s', algorithm_key, metric);
    if isfield(results, field_name)
        val = results.(field_name);
        if ~isempty(val) && isscalar(val) && isfinite(val)
            value = val;
        else
            value = 0; % 处理 NaN, Inf, [] 等情况
        end
    else
        value = 0; % 字段不存在时返回0
    end
end

%% 辅助函数 - 从智能体列表中提取防御者信息
function defenders = getDefenderInfo(agents)
    % 从agents单元数组中提取所有类型为'defender'的智能体信息。
    % 返回一个结构体数组，包含用于显示的名字和用于索引结果的键。
    defenders = struct('displayName', {}, 'key', {});
    for i = 1:length(agents)
        agent = agents{i};
        % 假设agent对象有 'type' 和 'name' 属性
        if isprop(agent, 'type') && strcmp(agent.type, 'defender')
            % 从 'QLearning防御者' 中提取 'QLearning'
            displayName = strrep(agent.name, '防御者', ''); 
            % 创建一个用于字段索引的key, e.g., 'DoubleQLearning' -> 'doubleqlearning'
            key = lower(strrep(displayName, ' ', '')); 
            
            info.displayName = displayName;
            info.key = key;
            defenders(end+1) = info;
        end
    end
end

%% 配置验证与默认值设置函数
function config = validateAndFixConfig(config)
    % 验证配置参数，为缺失的关键字段设置合理的默认值。
    % 核心作用：确保 n_episodes 是所有历史数据长度的唯一来源。
    
    if ~isfield(config, 'n_episodes')
        if isfield(config, 'n_iterations')
            config.n_episodes = config.n_iterations; % 兼容旧的 n_iterations
        else
            config.n_episodes = 500; % 设置一个安全的默认值
            fprintf('警告: 未找到 config.n_episodes 或 config.n_iterations, 已设为默认值 %d\n', config.n_episodes);
        end
    end
    
    % 定义并应用其他默认值
    defaults = {
        'n_stations', 10;
        'state_dim', 25
    };
    
    for i = 1:size(defaults, 1)
        if ~isfield(config, defaults{i,1})
            config.(defaults{i,1}) = defaults{i,2};
        end
    end
    
    % 动作空间维度依赖于站点数量
    if ~isfield(config, 'action_dim')
        config.action_dim = max(config.n_stations, 10);
    end
    
    % 确保报告目录存在
    if ~exist('reports', 'dir')
        mkdir('reports');
    end
end
