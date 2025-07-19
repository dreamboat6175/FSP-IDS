%% PerformanceVisualization.m - 增强的性能监控和可视化
% =========================================================================
% 描述: 提供实时监控和完整的可视化功能
% =========================================================================

classdef PerformanceVisualization < handle
    
    properties
        % 数据存储
        episode_data        % 每轮数据
        radi_history       % RADI历史
        damage_history     % 损害历史
        success_rate_history % 成功率历史
        attacker_strategy_history  % 攻击者策略历史
        defender_strategy_history  % 防御者策略历史
        
        % 配置
        config
        display_interval
        
        % 图形句柄
        figure_handle
        subplots
    end
    
    methods
        function obj = PerformanceVisualization(config)
            % 构造函数
            obj.config = config;
            obj.display_interval = config.display_interval;
            
            % 初始化数据存储
            obj.episode_data = [];
            obj.radi_history = [];
            obj.damage_history = [];
            obj.success_rate_history = [];
            obj.attacker_strategy_history = {};
            obj.defender_strategy_history = {};
            
            % 创建实时监控窗口
            obj.createMonitoringWindow();
        end
        
        function update(obj, episode, environment, agents, episode_results)
            % 更新监控数据
            
            % 提取当前数据
            current_radi = environment.radi_score;
            current_damage = mean(episode_results.damages);
            current_success_rate = mean(episode_results.attack_success);
            
            % 获取策略信息
            attacker_strategy = obj.extractStrategy(agents.attacker);
            defender_strategy = obj.extractStrategy(agents.defender);
            
            % 存储数据
            obj.radi_history(end+1) = current_radi;
            obj.damage_history(end+1) = current_damage;
            obj.success_rate_history(end+1) = current_success_rate;
            obj.attacker_strategy_history{end+1} = attacker_strategy;
            obj.defender_strategy_history{end+1} = defender_strategy;
            
            % 显示当前轮信息
            if mod(episode, obj.display_interval) == 0
                obj.displayEpisodeInfo(episode, current_radi, current_damage, ...
                    current_success_rate, attacker_strategy, defender_strategy);
                obj.updateRealTimeCharts(episode);
            end
        end
        
        function displayEpisodeInfo(obj, episode, radi, damage, success_rate, att_strategy, def_strategy)
            % 显示当前轮详细信息
            
            fprintf('\n========== Episode %d ==========\n', episode);
            fprintf('RADI Score: %.4f\n', radi);
            fprintf('Average Damage: %.4f\n', damage);
            fprintf('Attack Success Rate: %.2f%%\n', success_rate * 100);
            
            % 显示策略信息
            fprintf('\n攻击者策略:\n');
            obj.displayStrategy('攻击者', att_strategy);
            
            fprintf('\n防御者策略:\n');
            obj.displayStrategy('防御者', def_strategy);
            
            fprintf('================================\n');
        end
        
        function displayStrategy(obj, agent_name, strategy)
            % 显示策略详细信息
            
            if isstruct(strategy)
                if isfield(strategy, 'distribution')
                    fprintf('  策略分布: [');
                    fprintf('%.3f ', strategy.distribution);
                    fprintf(']\n');
                end
                if isfield(strategy, 'preferred_action')
                    fprintf('  首选动作: %d\n', strategy.preferred_action);
                end
                if isfield(strategy, 'exploration_rate')
                    fprintf('  探索率: %.3f\n', strategy.exploration_rate);
                end
            else
                fprintf('  策略: %s\n', mat2str(strategy));
            end
        end
        
        function strategy = extractStrategy(obj, agent)
            % 提取智能体策略信息
            
            strategy = struct();
            
            if isa(agent, 'QLearningAgent')
                % 获取Q表信息
                if isprop(agent, 'Q_table') && ~isempty(agent.Q_table)
                    % 计算策略分布（前5个状态的平均）
                    num_states = min(5, size(agent.Q_table, 1));
                    q_values = agent.Q_table(1:num_states, :);
                    
                    % 使用softmax计算策略分布
                    temperature = 0.1;
                    exp_q = exp(q_values / temperature);
                    strategy.distribution = mean(exp_q ./ sum(exp_q, 2), 1);
                    
                    % 获取最优动作
                    [~, preferred_actions] = max(q_values, [], 2);
                    strategy.preferred_action = mode(preferred_actions);
                end
            end
            
            % 添加探索率信息
            if isprop(agent, 'epsilon')
                strategy.exploration_rate = agent.epsilon;
            end
            
            % 添加学习率信息
            if isprop(agent, 'learning_rate')
                strategy.learning_rate = agent.learning_rate;
            end
        end
        
        function createMonitoringWindow(obj)
            % 创建实时监控窗口
            
            obj.figure_handle = figure('Name', 'FSP-TCS 实时监控', ...
                                      'Position', [100 100 1600 900], ...
                                      'NumberTitle', 'off');
            
            % 创建子图
            obj.subplots = struct();
            
            % 1. RADI演化
            obj.subplots.radi = subplot(3, 3, 1);
            title('RADI分数演化');
            xlabel('Episode');
            ylabel('RADI');
            grid on;
            hold on;
            
            % 2. 损害演化
            obj.subplots.damage = subplot(3, 3, 2);
            title('平均损害演化');
            xlabel('Episode');
            ylabel('损害');
            grid on;
            hold on;
            
            % 3. 成功率演化
            obj.subplots.success = subplot(3, 3, 3);
            title('攻击成功率演化');
            xlabel('Episode');
            ylabel('成功率 (%)');
            grid on;
            hold on;
            
            % 4. 攻击者策略分布
            obj.subplots.att_strategy = subplot(3, 3, 4);
            title('攻击者策略分布');
            xlabel('攻击类型');
            ylabel('概率');
            grid on;
            
            % 5. 防御者资源分配
            obj.subplots.def_strategy = subplot(3, 3, 5);
            title('防御者资源分配');
            xlabel('资源类型');
            ylabel('分配比例');
            grid on;
            
            % 6. 性能对比
            obj.subplots.comparison = subplot(3, 3, 6);
            title('攻防性能对比');
            xlabel('Episode');
            ylabel('累计奖励');
            grid on;
            hold on;
            
            % 7. 移动平均
            obj.subplots.moving_avg = subplot(3, 3, 7);
            title('性能移动平均（窗口=50）');
            xlabel('Episode');
            ylabel('指标值');
            grid on;
            hold on;
            
            % 8. 收敛性分析
            obj.subplots.convergence = subplot(3, 3, 8);
            title('收敛性分析');
            xlabel('Episode');
            ylabel('标准差');
            grid on;
            hold on;
            
            % 9. 实时统计
            obj.subplots.stats = subplot(3, 3, 9);
            title('实时统计信息');
            axis off;
        end
        
        function updateRealTimeCharts(obj, episode)
            % 更新实时图表
            
            episodes = 1:length(obj.radi_history);
            
            % 1. 更新RADI图
            subplot(obj.subplots.radi);
            cla;
            plot(episodes, obj.radi_history, 'b-', 'LineWidth', 2);
            title('RADI分数演化');
            xlabel('Episode');
            ylabel('RADI');
            grid on;
            
            % 2. 更新损害图
            subplot(obj.subplots.damage);
            cla;
            plot(episodes, obj.damage_history, 'r-', 'LineWidth', 2);
            title('平均损害演化');
            xlabel('Episode');
            ylabel('损害');
            grid on;
            
            % 3. 更新成功率图
            subplot(obj.subplots.success);
            cla;
            plot(episodes, obj.success_rate_history * 100, 'g-', 'LineWidth', 2);
            title('攻击成功率演化');
            xlabel('Episode');
            ylabel('成功率 (%)');
            grid on;
            
            % 4. 更新攻击者策略分布
            if ~isempty(obj.attacker_strategy_history)
                subplot(obj.subplots.att_strategy);
                cla;
                latest_strategy = obj.attacker_strategy_history{end};
                if isfield(latest_strategy, 'distribution')
                    bar(latest_strategy.distribution);
                    title('攻击者策略分布');
                    xlabel('攻击类型');
                    ylabel('概率');
                    grid on;
                end
            end
            
            % 5. 更新防御者资源分配
            if ~isempty(obj.defender_strategy_history)
                subplot(obj.subplots.def_strategy);
                cla;
                latest_strategy = obj.defender_strategy_history{end};
                if isfield(latest_strategy, 'distribution')
                    bar(latest_strategy.distribution);
                    title('防御者资源分配');
                    xlabel('资源类型');
                    ylabel('分配比例');
                    grid on;
                end
            end
            
            % 7. 更新移动平均
            subplot(obj.subplots.moving_avg);
            cla;
            window = 50;
            if length(obj.radi_history) >= window
                ma_radi = movmean(obj.radi_history, window);
                ma_success = movmean(obj.success_rate_history, window);
                plot(episodes, ma_radi, 'b-', 'LineWidth', 2);
                hold on;
                plot(episodes, ma_success, 'g-', 'LineWidth', 2);
                legend('RADI移动平均', '成功率移动平均');
                title(sprintf('%d-Episode移动平均', window));
                xlabel('Episode');
                ylabel('指标值');
                grid on;
            end
            
            % 8. 更新收敛性分析
            subplot(obj.subplots.convergence);
            cla;
            window = 50;
            if length(obj.radi_history) >= window
                radi_std = movstd(obj.radi_history, window);
                success_std = movstd(obj.success_rate_history, window);
                plot(window:length(radi_std), radi_std(window:end), 'b-', 'LineWidth', 2);
                hold on;
                plot(window:length(success_std), success_std(window:end), 'g-', 'LineWidth', 2);
                legend('RADI标准差', '成功率标准差');
                title('收敛性分析（滑动窗口标准差）');
                xlabel('Episode');
                ylabel('标准差');
                grid on;
            end
            
            % 9. 更新统计信息
            subplot(obj.subplots.stats);
            cla;
            axis off;
            stats_text = sprintf([...
                '当前Episode: %d\n\n' ...
                '最新指标:\n' ...
                '  RADI: %.4f\n' ...
                '  损害: %.4f\n' ...
                '  成功率: %.2f%%\n\n' ...
                '历史最优:\n' ...
                '  最高RADI: %.4f\n' ...
                '  最低损害: %.4f\n' ...
                '  最高成功率: %.2f%%\n\n' ...
                '收敛指标:\n' ...
                '  RADI方差: %.4f\n' ...
                '  成功率方差: %.4f'], ...
                episode, ...
                obj.radi_history(end), ...
                obj.damage_history(end), ...
                obj.success_rate_history(end) * 100, ...
                max(obj.radi_history), ...
                min(obj.damage_history), ...
                max(obj.success_rate_history) * 100, ...
                var(obj.radi_history(max(1,end-49):end)), ...
                var(obj.success_rate_history(max(1,end-49):end)));
            text(0.1, 0.5, stats_text, 'FontSize', 10, 'FontName', 'Courier');
            
            % 刷新图形
            drawnow;
        end
        
        function generateFinalReport(obj, save_path)
            % 生成最终的可视化报告
            
            figure('Position', [50 50 1800 1200]);
            
            % 设置总标题
            sgtitle('FSP-TCS 仿真结果报告', 'FontSize', 16, 'FontWeight', 'bold');
            
            episodes = 1:length(obj.radi_history);
            
            % 1. RADI演化曲线
            subplot(4, 4, [1 2]);
            plot(episodes, obj.radi_history, 'b-', 'LineWidth', 2);
            hold on;
            if length(episodes) > 50
                ma_radi = movmean(obj.radi_history, 50);
                plot(episodes, ma_radi, 'r--', 'LineWidth', 2);
                legend('RADI', '50-Episode MA');
            end
            title('RADI分数演化');
            xlabel('Episode');
            ylabel('RADI Score');
            grid on;
            
            % 2. 攻击成功率演化
            subplot(4, 4, [3 4]);
            plot(episodes, obj.success_rate_history * 100, 'g-', 'LineWidth', 2);
            hold on;
            if length(episodes) > 50
                ma_success = movmean(obj.success_rate_history * 100, 50);
                plot(episodes, ma_success, 'r--', 'LineWidth', 2);
                legend('成功率', '50-Episode MA');
            end
            title('攻击成功率演化');
            xlabel('Episode');
            ylabel('成功率 (%)');
            grid on;
            
            % 3. 损害分布
            subplot(4, 4, [5 6]);
            histogram(obj.damage_history, 30, 'FaceColor', 'r', 'EdgeColor', 'darkred');
            title('损害值分布');
            xlabel('损害值');
            ylabel('频次');
            grid on;
            
            % 4. 策略演化热力图
            subplot(4, 4, [7 8]);
            if length(obj.attacker_strategy_history) > 10
                strategy_matrix = [];
                sample_indices = round(linspace(1, length(obj.attacker_strategy_history), 20));
                for i = sample_indices
                    if isfield(obj.attacker_strategy_history{i}, 'distribution')
                        strategy_matrix = [strategy_matrix; obj.attacker_strategy_history{i}.distribution];
                    end
                end
                if ~isempty(strategy_matrix)
                    imagesc(strategy_matrix');
                    colormap('hot');
                    colorbar;
                    title('攻击者策略演化热力图');
                    xlabel('Episode (采样)');
                    ylabel('攻击类型');
                end
            end
            
            % 5. 性能指标箱线图
            subplot(4, 4, [9 10]);
            data_matrix = [obj.radi_history', obj.damage_history', obj.success_rate_history'];
            boxplot(data_matrix, {'RADI', '损害', '成功率'});
            title('性能指标分布');
            ylabel('指标值');
            grid on;
            
            % 6. 收敛性分析
            subplot(4, 4, [11 12]);
            window_sizes = [10, 20, 50, 100];
            colors = {'b', 'r', 'g', 'm'};
            hold on;
            for i = 1:length(window_sizes)
                if length(obj.radi_history) >= window_sizes(i)
                    conv_metric = movstd(obj.radi_history, window_sizes(i));
                    plot(window_sizes(i):length(conv_metric), conv_metric(window_sizes(i):end), ...
                         colors{i}, 'LineWidth', 1.5);
                end
            end
            legend(arrayfun(@(x) sprintf('窗口=%d', x), window_sizes, 'UniformOutput', false));
            title('RADI收敛性分析（滑动标准差）');
            xlabel('Episode');
            ylabel('标准差');
            grid on;
            
            % 7. 相关性分析
            subplot(4, 4, 13);
            if length(obj.radi_history) > 10
                scatter(obj.success_rate_history, obj.radi_history, 20, episodes, 'filled');
                colormap('jet');
                colorbar;
                xlabel('攻击成功率');
                ylabel('RADI分数');
                title('成功率 vs RADI（颜色=Episode）');
                grid on;
            end
            
            % 8. 学习曲线对比
            subplot(4, 4, 14);
            segments = 5;
            segment_size = floor(length(episodes) / segments);
            segment_means_radi = [];
            segment_means_success = [];
            for i = 1:segments
                start_idx = (i-1) * segment_size + 1;
                end_idx = min(i * segment_size, length(episodes));
                segment_means_radi(i) = mean(obj.radi_history(start_idx:end_idx));
                segment_means_success(i) = mean(obj.success_rate_history(start_idx:end_idx));
            end
            x = 1:segments;
            yyaxis left;
            bar(x, segment_means_radi, 'b');
            ylabel('平均RADI');
            yyaxis right;
            plot(x, segment_means_success * 100, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
            ylabel('平均成功率 (%)');
            xlabel('训练阶段');
            title('分阶段性能对比');
            grid on;
            
            % 9. 最终统计表
            subplot(4, 4, [15 16]);
            axis off;
            stats_table = {
                '指标', '最小值', '最大值', '平均值', '标准差', '最终值';
                'RADI', sprintf('%.4f', min(obj.radi_history)), ...
                        sprintf('%.4f', max(obj.radi_history)), ...
                        sprintf('%.4f', mean(obj.radi_history)), ...
                        sprintf('%.4f', std(obj.radi_history)), ...
                        sprintf('%.4f', obj.radi_history(end));
                '损害', sprintf('%.4f', min(obj.damage_history)), ...
                        sprintf('%.4f', max(obj.damage_history)), ...
                        sprintf('%.4f', mean(obj.damage_history)), ...
                        sprintf('%.4f', std(obj.damage_history)), ...
                        sprintf('%.4f', obj.damage_history(end));
                '成功率(%)', sprintf('%.2f', min(obj.success_rate_history)*100), ...
                             sprintf('%.2f', max(obj.success_rate_history)*100), ...
                             sprintf('%.2f', mean(obj.success_rate_history)*100), ...
                             sprintf('%.2f', std(obj.success_rate_history)*100), ...
                             sprintf('%.2f', obj.success_rate_history(end)*100);
            };
            
            % 创建表格
            table_pos = [0.1, 0.3, 0.8, 0.4];
            uitable('Data', stats_table, ...
                   'Units', 'normalized', ...
                   'Position', table_pos, ...
                   'FontSize', 10);
            title('性能统计汇总');
            
            % 保存图形
            if nargin > 1 && ~isempty(save_path)
                saveas(gcf, save_path);
                fprintf('可视化报告已保存到: %s\n', save_path);
            end
        end
    end
end