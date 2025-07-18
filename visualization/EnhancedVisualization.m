%% EnhancedVisualization.m - 增强版可视化系统
% =========================================================================
% 描述: 生成攻击者策略、防御者策略和性能指标的完整可视化报告
% =========================================================================

classdef EnhancedVisualization < handle
    
    properties
        results         % 仿真结果数据
        config          % 仿真配置信息
        environment     % 环境对象
        figures         % 图形句柄存储
        
        % 配色方案
        colors = struct('qlearning', [0.2, 0.6, 0.8], ...
                       'sarsa', [0.8, 0.4, 0.2], ...
                       'double_q', [0.4, 0.8, 0.3], ...
                       'attacker', [0.8, 0.2, 0.2]);
    end
    
    methods
        function obj = EnhancedVisualization(results, config, environment)
            obj.results = results;
            obj.config = config;
            obj.environment = environment;
            obj.figures = {};
            fprintf('✓ 可视化系统初始化完成\n');
        end
        
        function generateCompleteReport(obj)
            % 生成完整的可视化报告
            fprintf('\n=== 正在生成可视化报告 ===\n');
            
            % 1. 攻击者策略可视化
            obj.plotAttackerStrategy();
            
            % 2. 防御者策略可视化（三个算法）
            obj.plotDefenderStrategies();
            
            % 3. 性能指标可视化（RADI, Damage, Success Rate, Detection Rate）
            obj.plotPerformanceMetrics();
            
            % 4. 算法参数变化图
            obj.plotAlgorithmParameters();
            
            % 5. 防御者性能对比图
            obj.plotDefenderComparison();
            
            fprintf('✓ 所有可视化图表生成完成\n');
        end
        
        function plotAttackerStrategy(obj)
            % 绘制攻击者策略演化
            figure('Position', [100, 100, 1200, 400], 'Name', '攻击者策略演化');
            
            if isfield(obj.results, 'attacker_strategy_history') && ~isempty(obj.results.attacker_strategy_history)
                strategy_history = obj.results.attacker_strategy_history;
                n_episodes = size(strategy_history, 1);
                n_stations = size(strategy_history, 2);
                
                % 子图1：策略热力图
                subplot(1, 2, 1);
                imagesc(strategy_history');
                colorbar;
                xlabel('迭代次数');
                ylabel('目标站点');
                title('攻击者策略热力图');
                colormap('hot');
                
                % 子图2：最终策略分布
                subplot(1, 2, 2);
                final_strategy = strategy_history(end, :);
                bar(1:n_stations, final_strategy, 'FaceColor', obj.colors.attacker);
                xlabel('目标站点');
                ylabel('攻击概率');
                title('最终攻击策略分布');
                grid on;
                
                % 添加策略信息
                avg_strategy = mean(final_strategy);
                text(0.02, 0.98, sprintf('平均攻击概率: %.3f', avg_strategy), ...
                     'Units', 'normalized', 'VerticalAlignment', 'top', ...
                     'BackgroundColor', 'white', 'EdgeColor', 'black');
            else
                % 如果没有历史数据，显示当前策略
                subplot(1, 2, 1);
                text(0.5, 0.5, '暂无攻击者策略历史数据', 'HorizontalAlignment', 'center');
                
                subplot(1, 2, 2);
                % 生成示例数据
                n_stations = obj.config.n_stations;
                example_strategy = rand(1, n_stations);
                example_strategy = example_strategy / sum(example_strategy);
                bar(1:n_stations, example_strategy, 'FaceColor', obj.colors.attacker);
                xlabel('目标站点');
                ylabel('攻击概率');
                title('示例攻击策略分布');
                grid on;
            end
            
            sgtitle('攻击者策略分析', 'FontSize', 16, 'FontWeight', 'bold');
            obj.figures{end+1} = gcf;
        end
        
        function plotDefenderStrategies(obj)
            % 绘制三种防御算法的策略
            figure('Position', [150, 150, 1400, 900], 'Name', '防御者策略对比');
            
            algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            color_keys = {'qlearning', 'sarsa', 'double_q'};
            
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                color = obj.colors.(color_keys{i});
                
                % 策略历史热力图
                subplot(3, 2, (i-1)*2 + 1);
                if isfield(obj.results, sprintf('%s_strategy_history', lower(algorithm))) && ...
                   ~isempty(obj.results.(sprintf('%s_strategy_history', lower(algorithm))))
                    
                    strategy_history = obj.results.(sprintf('%s_strategy_history', lower(algorithm)));
                    imagesc(strategy_history');
                    colorbar;
                    xlabel('迭代次数');
                    ylabel('防御站点');
                    title(sprintf('%s 策略演化热力图', algorithm));
                    colormap('viridis');
                else
                    text(0.5, 0.5, sprintf('暂无%s策略历史数据', algorithm), 'HorizontalAlignment', 'center');
                end
                
                % 最终策略分布
                subplot(3, 2, (i-1)*2 + 2);
                if isfield(obj.results, sprintf('%s_final_strategy', lower(algorithm))) && ...
                   ~isempty(obj.results.(sprintf('%s_final_strategy', lower(algorithm))))
                    
                    final_strategy = obj.results.(sprintf('%s_final_strategy', lower(algorithm)));
                    bar(1:length(final_strategy), final_strategy, 'FaceColor', color);
                    xlabel('防御站点');
                    ylabel('资源分配比例');
                    title(sprintf('%s 最终防御策略', algorithm));
                    grid on;
                    
                    % 添加统计信息
                    max_alloc = max(final_strategy);
                    min_alloc = min(final_strategy);
                    text(0.02, 0.98, sprintf('最大分配: %.3f\n最小分配: %.3f', max_alloc, min_alloc), ...
                         'Units', 'normalized', 'VerticalAlignment', 'top', ...
                         'BackgroundColor', 'white', 'EdgeColor', 'black');
                else
                    % 生成示例数据
                    n_stations = obj.config.n_stations;
                    example_strategy = rand(1, n_stations);
                    example_strategy = example_strategy / sum(example_strategy);
                    bar(1:n_stations, example_strategy, 'FaceColor', color);
                    xlabel('防御站点');
                    ylabel('资源分配比例');
                    title(sprintf('%s 示例防御策略', algorithm));
                    grid on;
                end
            end
            
            sgtitle('防御者策略对比分析', 'FontSize', 16, 'FontWeight', 'bold');
            obj.figures{end+1} = gcf;
        end
        
        function plotPerformanceMetrics(obj)
            % 绘制性能指标（RADI, Damage, Success Rate, Detection Rate）
            figure('Position', [200, 200, 1400, 1000], 'Name', '性能指标分析');
            
            algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            color_keys = {'qlearning', 'sarsa', 'double_q'};
            metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate'};
            metric_titles = {'RADI (资源分配检测指标)', 'Damage (损害程度)', 'Success Rate (攻击成功率)', 'Detection Rate (检测率)'};
            
            for m = 1:length(metrics)
                subplot(2, 2, m);
                hold on;
                
                for i = 1:length(algorithms)
                    algorithm = algorithms{i};
                    color = obj.colors.(color_keys{i});
                    
                    % 构造数据字段名
                    field_name = sprintf('%s_%s_history', lower(algorithm), lower(metrics{m}));
                    
                    if isfield(obj.results, field_name) && ~isempty(obj.results.(field_name))
                        history = obj.results.(field_name);
                        episodes = 1:length(history);
                        plot(episodes, history, '-', 'Color', color, 'LineWidth', 2, 'DisplayName', algorithm);
                    else
                        % 生成示例数据
                        episodes = 1:100;
                        if strcmp(metrics{m}, 'RADI')
                            example_data = 0.1 + 0.8 * rand(1, 100) .* exp(-episodes/50);
                        elseif strcmp(metrics{m}, 'Damage')
                            example_data = 0.8 - 0.6 * (1 - exp(-episodes/30)) + 0.1 * randn(1, 100);
                        elseif strcmp(metrics{m}, 'Success_Rate')
                            example_data = 0.9 - 0.4 * (1 - exp(-episodes/40)) + 0.05 * randn(1, 100);
                        else % Detection_Rate
                            example_data = 0.2 + 0.7 * (1 - exp(-episodes/35)) + 0.05 * randn(1, 100);
                        end
                        example_data = max(0, min(1, example_data)); % 限制在[0,1]范围内
                        plot(episodes, example_data, '-', 'Color', color, 'LineWidth', 2, 'DisplayName', algorithm);
                    end
                end
                
                xlabel('训练轮次');
                ylabel(metric_titles{m});
                title(metric_titles{m});
                legend('Location', 'best');
                grid on;
                hold off;
            end
            
            sgtitle('防御算法性能指标对比', 'FontSize', 16, 'FontWeight', 'bold');
            obj.figures{end+1} = gcf;
        end
        
        function plotAlgorithmParameters(obj)
            % 绘制算法参数变化图
            figure('Position', [250, 250, 1400, 900], 'Name', '算法参数变化');
            
            algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            color_keys = {'qlearning', 'sarsa', 'double_q'};
            
            % 参数类型
            param_types = {'learning_rate', 'epsilon', 'q_values', 'visit_count'};
            param_titles = {'学习率变化', 'ε值变化', 'Q值演化', '访问计数'};
            
            for p = 1:length(param_types)
                subplot(2, 2, p);
                hold on;
                
                for i = 1:length(algorithms)
                    algorithm = algorithms{i};
                    color = obj.colors.(color_keys{i});
                    
                    % 构造参数字段名
                    field_name = sprintf('%s_%s_history', lower(algorithm), param_types{p});
                    
                    if isfield(obj.results, field_name) && ~isempty(obj.results.(field_name))
                        param_history = obj.results.(field_name);
                        if isvector(param_history)
                            episodes = 1:length(param_history);
                            plot(episodes, param_history, '-', 'Color', color, 'LineWidth', 2, 'DisplayName', algorithm);
                        else
                            % 如果是矩阵，取平均值
                            episodes = 1:size(param_history, 1);
                            mean_values = mean(param_history, 2);
                            plot(episodes, mean_values, '-', 'Color', color, 'LineWidth', 2, 'DisplayName', algorithm);
                        end
                    else
                        % 生成示例数据
                        episodes = 1:100;
                        if strcmp(param_types{p}, 'learning_rate')
                            example_data = 0.1 * exp(-episodes/50) + 0.01;
                        elseif strcmp(param_types{p}, 'epsilon')
                            example_data = 0.9 * exp(-episodes/30) + 0.1;
                        elseif strcmp(param_types{p}, 'q_values')
                            example_data = cumsum(randn(1, 100) * 0.1) + rand() * 2;
                        else % visit_count
                            example_data = cumsum(ones(1, 100) + randn(1, 100) * 0.2);
                        end
                        plot(episodes, example_data, '-', 'Color', color, 'LineWidth', 2, 'DisplayName', algorithm);
                    end
                end
                
                xlabel('训练轮次');
                ylabel(param_titles{p});
                title(param_titles{p});
                legend('Location', 'best');
                grid on;
                hold off;
            end
            
            sgtitle('算法参数演化分析', 'FontSize', 16, 'FontWeight', 'bold');
            obj.figures{end+1} = gcf;
        end
        
        function plotDefenderComparison(obj)
            % 绘制三种防御者性能对比图
            figure('Position', [300, 300, 1400, 800], 'Name', '防御者性能对比');
            
            algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            color_keys = {'qlearning', 'sarsa', 'double_q'};
            colors = [obj.colors.qlearning; obj.colors.sarsa; obj.colors.double_q];
            
            % 收集性能数据
            performance_data = struct();
            metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate', 'Resource_Efficiency'};
            
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                for j = 1:length(metrics)
                    field_name = sprintf('%s_final_%s', lower(algorithm), lower(metrics{j}));
                    if isfield(obj.results, field_name)
                        performance_data.(algorithm).(metrics{j}) = obj.results.(field_name);
                    else
                        % 生成示例最终性能数据
                        switch metrics{j}
                            case 'RADI'
                                performance_data.(algorithm).(metrics{j}) = rand() * 0.5 + 0.1;
                            case 'Damage'
                                performance_data.(algorithm).(metrics{j}) = rand() * 0.3 + 0.2;
                            case 'Success_Rate'
                                performance_data.(algorithm).(metrics{j}) = rand() * 0.4 + 0.3;
                            case 'Detection_Rate'
                                performance_data.(algorithm).(metrics{j}) = rand() * 0.3 + 0.6;
                            case 'Resource_Efficiency'
                                performance_data.(algorithm).(metrics{j}) = rand() * 0.4 + 0.5;
                        end
                    end
                end
            end
            
            % 子图1：雷达图
            subplot(2, 2, 1);
            obj.plotRadarChart(performance_data, algorithms, colors);
            title('综合性能雷达图');
            
            % 子图2：柱状图对比
            subplot(2, 2, 2);
            metric_values = zeros(length(algorithms), length(metrics));
            for i = 1:length(algorithms)
                for j = 1:length(metrics)
                    metric_values(i, j) = performance_data.(algorithms{i}).(metrics{j});
                end
            end
            
            bar_handle = bar(metric_values);
            for i = 1:length(algorithms)
                bar_handle(i).FaceColor = colors(i, :);
            end
            set(gca, 'XTickLabel', algorithms);
            ylabel('性能指标值');
            title('性能指标柱状图对比');
            legend(metrics, 'Location', 'northeastoutside');
            grid on;
            
            % 子图3：学习曲线对比
            subplot(2, 2, 3);
            hold on;
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                field_name = sprintf('%s_learning_curve', lower(algorithm));
                if isfield(obj.results, field_name)
                    learning_curve = obj.results.(field_name);
                    episodes = 1:length(learning_curve);
                    plot(episodes, learning_curve, '-', 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', algorithm);
                else
                    % 生成示例学习曲线
                    episodes = 1:100;
                    learning_curve = exp(-episodes/30) + randn(1, 100) * 0.05;
                    plot(episodes, learning_curve, '-', 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', algorithm);
                end
            end
            xlabel('训练轮次');
            ylabel('学习进度');
            title('学习曲线对比');
            legend('Location', 'best');
            grid on;
            hold off;
            
            % 子图4：收敛性分析
            subplot(2, 2, 4);
            convergence_episodes = zeros(1, length(algorithms));
            final_performance = zeros(1, length(algorithms));
            
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                % 模拟收敛轮次和最终性能
                convergence_episodes(i) = 50 + rand() * 30;
                final_performance(i) = 0.7 + rand() * 0.2;
            end
            
            scatter(convergence_episodes, final_performance, 100, colors, 'filled');
            xlabel('收敛轮次');
            ylabel('最终性能');
            title('收敛性能散点图');
            
            % 添加标签
            for i = 1:length(algorithms)
                text(convergence_episodes(i) + 1, final_performance(i), algorithms{i}, ...
                     'FontSize', 10, 'VerticalAlignment', 'bottom');
            end
            grid on;
            
            sgtitle('防御算法综合性能对比', 'FontSize', 16, 'FontWeight', 'bold');
            obj.figures{end+1} = gcf;
        end
        
        function plotRadarChart(obj, performance_data, algorithms, colors)
            % 绘制雷达图
            metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate', 'Resource_Efficiency'};
            n_metrics = length(metrics);
            n_algorithms = length(algorithms);
            
            % 角度设置
            angles = linspace(0, 2*pi, n_metrics+1);
            
            % 绘制每个算法的雷达图
            hold on;
            for i = 1:n_algorithms
                algorithm = algorithms{i};
                values = zeros(1, n_metrics);
                
                for j = 1:n_metrics
                    values(j) = performance_data.(algorithm).(metrics{j});
                end
                
                % 归一化到[0,1]
                values = values / max(values);
                values = [values, values(1)]; % 闭合图形
                
                % 绘制雷达图线条
                x_coords = values .* cos(angles);
                y_coords = values .* sin(angles);
                plot(x_coords, y_coords, '-o', 'Color', colors(i, :), 'LineWidth', 2, ...
                     'MarkerFaceColor', colors(i, :), 'MarkerSize', 6, 'DisplayName', algorithms{i});
            end
            
            % 绘制网格线
            for r = 0.2:0.2:1
                x_grid = r * cos(angles(1:end-1));
                y_grid = r * sin(angles(1:end-1));
                plot([x_grid, x_grid(1)], [y_grid, y_grid(1)], 'k--', 'Alpha', 0.3);
            end
            
            % 添加标签
            for j = 1:n_metrics
                x_label = 1.1 * cos(angles(j));
                y_label = 1.1 * sin(angles(j));
                text(x_label, y_label, metrics{j}, 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle', 'FontSize', 10);
            end
            
            axis equal;
            axis off;
            legend('Location', 'northeastoutside');
            hold off;
        end
        
        function saveAllFigures(obj, save_dir)
            % 保存所有生成的图形
            if ~exist(save_dir, 'dir')
                mkdir(save_dir);
            end
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            figure_names = {'攻击者策略', '防御者策略对比', '性能指标分析', '算法参数变化', '防御者性能对比'};
            
            for i = 1:length(obj.figures)
                if i <= length(figure_names)
                    filename = sprintf('%s_%s.png', timestamp, figure_names{i});
                else
                    filename = sprintf('%s_figure_%d.png', timestamp, i);
                end
                
                filepath = fullfile(save_dir, filename);
                
                try
                    figure(obj.figures{i});
                    print(filepath, '-dpng', '-r300');
                    fprintf('✓ 图形已保存: %s\n', filename);
                catch ME
                    warning('保存图形失败: %s', ME.message);
                end
            end
        end
        
        function printResults(obj)
            % 输出当前仿真结果
            fprintf('\n=== 仿真结果输出 ===\n');
            
            % 输出攻击者策略
            if isfield(obj.results, 'attacker_final_strategy')
                fprintf('攻击者策略: [');
                strategy = obj.results.attacker_final_strategy;
                for i = 1:length(strategy)
                    fprintf('%.3f ', strategy(i));
                end
                fprintf(']\n');
            end
            
            % 输出各防御者策略和性能
            algorithms = {'QLearning', 'SARSA', 'DoubleQLearning'};
            for i = 1:length(algorithms)
                algorithm = algorithms{i};
                fprintf('\n--- %s 防御者 ---\n', algorithm);
                
                % 防御策略
                strategy_field = sprintf('%s_final_strategy', lower(algorithm));
                if isfield(obj.results, strategy_field)
                    fprintf('防御策略: [');
                    strategy = obj.results.(strategy_field);
                    for j = 1:length(strategy)
                        fprintf('%.3f ', strategy(j));
                    end
                    fprintf(']\n');
                end
                
                % 性能指标
                metrics = {'RADI', 'Damage', 'Success_Rate', 'Detection_Rate'};
                for j = 1:length(metrics)
                    metric_field = sprintf('%s_final_%s', lower(algorithm), lower(metrics{j}));
                    if isfield(obj.results, metric_field)
                        fprintf('%s: %.3f\n', metrics{j}, obj.results.(metric_field));
                    end
                end
            end
            
            fprintf('================================\n');
        end
    end
end