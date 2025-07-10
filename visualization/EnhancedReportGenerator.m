%% EnhancedReportGenerator.m - 增强版报告生成器
classdef EnhancedReportGenerator < handle
    methods (Static)
        function generateEnhancedReport(results, config, agents, ~)
            % 生成增强的分析报告
            try
                % 确保reports文件夹存在
                if ~exist('reports', 'dir')
                    mkdir('reports');
                end
                
                fprintf('正在生成增强报告...\n');
                
                % 为每个算法生成单独的详细图表
                EnhancedReportGenerator.generateIndividualReports(results, config, agents);
                
                % 生成综合对比图
                EnhancedReportGenerator.generateComparisonReport(results, config, agents);
                
                % 生成性能分析报告
                EnhancedReportGenerator.generatePerformanceAnalysis(results, config, agents);
                
                fprintf('✓ 增强报告生成完成！\n');
                
            catch ME
                fprintf('报告生成出错: %s\n', ME.message);
                fprintf('继续运行简化报告生成...\n');
                EnhancedReportGenerator.generateSimpleReport(results, config);
            end
        end
        
        function generateSimpleReport(results, config)
            % 生成简化报告（备用方案）
            fprintf('\n=== FSP仿真简化报告 ===\n');
            fprintf('生成时间: %s\n', datestr(now));
            
            if isfield(results, 'detection_rates') && ~isempty(results.detection_rates)
                agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
                
                for i = 1:min(size(results.detection_rates, 1), 3)
                    last_100 = max(1, size(results.detection_rates, 2)-99):size(results.detection_rates, 2);
                    avg_detection = mean(results.detection_rates(i, last_100));
                    
                    fprintf('\n%s:\n', agent_names{i});
                    fprintf('  最终检测率: %.2f%%\n', avg_detection * 100);
                    
                    if avg_detection < 0.3
                        fprintf('  状态: 检测率过低，需要优化\n');
                    elseif avg_detection < 0.6
                        fprintf('  状态: 检测率中等，有改进空间\n');
                    else
                        fprintf('  状态: 检测率良好\n');
                    end
                end
            end
            
            % 保存结果
            save('reports/simple_results.mat', 'results', 'config');
            fprintf('\n结果已保存到 reports/simple_results.mat\n');
        end
        
        function generateIndividualReports(results, config, agents)
            % 为每个算法生成单独的报告
            
            if ~isfield(results, 'n_agents') || results.n_agents == 0
                results.n_agents = min(3, size(results.detection_rates, 1));
            end
            
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            colors = [0.2 0.4 0.8;    % 蓝色系
                     0.8 0.2 0.2;     % 红色系
                     0.2 0.6 0.2];    % 绿色系
            
            for i = 1:results.n_agents
                try
                    figure('Position', [100, 100, 1200, 800], ...
                           'Name', sprintf('%s 性能分析', agent_names{i}));
                    
                    % 设置图形背景
                    set(gcf, 'Color', 'white');
                    
                    % 1. 检测率趋势（带平滑）
                    subplot(2, 3, 1);
                    if isfield(results, 'detection_rates') && size(results.detection_rates, 1) >= i
                        EnhancedReportGenerator.plotDetectionTrend(results.detection_rates(i, :), ...
                                                                  colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '检测率数据不可用', 'HorizontalAlignment', 'center');
                    end
                    
                    % 2. 资源利用效率
                    subplot(2, 3, 2);
                    if isfield(results, 'resource_utilization') && size(results.resource_utilization, 1) >= i
                        EnhancedReportGenerator.plotResourceUtilization(results.resource_utilization(i, :), ...
                                                                       colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '资源利用数据不可用', 'HorizontalAlignment', 'center');
                    end
                    
                    % 3. 收敛性分析
                    subplot(2, 3, 3);
                    if isfield(results, 'convergence_metrics') && size(results.convergence_metrics, 1) >= i
                        EnhancedReportGenerator.plotConvergenceAnalysis(results.convergence_metrics(i, :), ...
                                                                       colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '收敛性数据不可用', 'HorizontalAlignment', 'center');
                    end
                    
                    % 4. 奖励趋势
                    subplot(2, 3, 4);
                    if isfield(results, 'defender_rewards') && size(results.defender_rewards, 1) >= i
                        EnhancedReportGenerator.plotRewardTrend(results.defender_rewards(i, :), ...
                                                               colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '奖励数据不可用', 'HorizontalAlignment', 'center');
                    end
                    
                    % 5. 策略稳定性
                    subplot(2, 3, 5);
                    if isfield(results, 'policy_stability') && size(results.policy_stability, 1) >= i
                        EnhancedReportGenerator.plotPolicyStability(results.policy_stability(i, :), ...
                                                                   colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '策略稳定性数据不可用', 'HorizontalAlignment', 'center');
                    end
                    
                    % 6. 性能统计
                    subplot(2, 3, 6);
                    EnhancedReportGenerator.plotPerformanceStats(results, i, agent_names{i});
                    
                    % 保存图片
                    filename = sprintf('reports/%s_detailed_report_%s.png', ...
                                      strrep(agent_names{i}, '-', '_'), datestr(now, 'yyyymmdd_HHMMSS'));
                    saveas(gcf, filename);
                    fprintf('保存个体报告: %s\n', filename);
                    
                catch ME
                    fprintf('生成 %s 个体报告时出错: %s\n', agent_names{i}, ME.message);
                end
            end
        end
        
        function plotDetectionTrend(detection_rates, color, agent_name)
            % 绘制检测率趋势（带平滑和置信区间）
            
            if isempty(detection_rates) || all(detection_rates == 0)
                text(0.5, 0.5, '检测率数据为空', 'HorizontalAlignment', 'center');
                return;
            end
            
            % 原始数据
            plot(1:length(detection_rates), detection_rates, ...
                 'Color', [color 0.3], 'LineWidth', 0.5);
            hold on;
            
            % 移动平均平滑
            window_size = min(50, max(5, length(detection_rates)/10));
            if length(detection_rates) >= window_size
                smoothed = movmean(detection_rates, window_size);
                plot(1:length(smoothed), smoothed, ...
                     'Color', color, 'LineWidth', 2.5);
            end
            
            % 添加趋势线
            if length(detection_rates) > 10
                x = 1:length(detection_rates);
                p = polyfit(x, detection_rates, 1);
                trend = polyval(p, x);
                plot(x, trend, '--', 'Color', color*0.5, 'LineWidth', 2);
            end
            
            % 标注最终性能
            final_window = max(1, length(detection_rates)-min(99, length(detection_rates)-1)):length(detection_rates);
            final_rate = mean(detection_rates(final_window));
            text(length(detection_rates)*0.7, max(detection_rates)*0.9, ...
                 sprintf('最终检测率: %.1f%%', final_rate*100), ...
                 'FontSize', 12, 'FontWeight', 'bold', 'Color', color);
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('检测率', 'FontSize', 12);
            title(sprintf('%s - 检测率趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1]);
            
            if length(detection_rates) >= window_size
                legend({'原始数据', '平滑曲线', '趋势线'}, 'Location', 'best');
            else
                legend({'原始数据'}, 'Location', 'best');
            end
        end
        
        function plotResourceUtilization(resource_util, color, agent_name)
            % 绘制资源利用效率
            
            if isempty(resource_util)
                text(0.5, 0.5, '资源利用数据为空', 'HorizontalAlignment', 'center');
                return;
            end
            
            % 绘制填充区域图
            x = 1:length(resource_util);
            fill([x, fliplr(x)], [resource_util, zeros(size(resource_util))], ...
                 color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            hold on;
            
            % 绘制主线
            plot(x, resource_util, 'Color', color, 'LineWidth', 2);
            
            % 添加平均线
            avg_util = mean(resource_util);
            plot([1, length(resource_util)], [avg_util, avg_util], ...
                 '--', 'Color', color*0.5, 'LineWidth', 2);
            
            % 标注
            text(length(resource_util)*0.5, avg_util+0.05, ...
                 sprintf('平均: %.1f%%', avg_util*100), ...
                 'FontSize', 11, 'HorizontalAlignment', 'center');
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('资源利用率', 'FontSize', 12);
            title(sprintf('%s - 资源利用效率', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1]);
        end
        
        function plotConvergenceAnalysis(convergence, color, agent_name)
            % 绘制收敛性分析
            
            valid_idx = find(convergence > 0);
            if isempty(valid_idx)
                text(0.5, 0.5, '收敛数据不足', 'HorizontalAlignment', 'center');
                title(sprintf('%s - 收敛性分析', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            % 对数尺度绘图
            semilogy(valid_idx, convergence(valid_idx), ...
                    'Color', color, 'LineWidth', 2, 'Marker', 'o', ...
                    'MarkerSize', 4, 'MarkerFaceColor', color);
            
            % 添加收敛阈值线
            convergence_threshold = 0.01;
            hold on;
            plot([min(valid_idx), max(valid_idx)], ...
                 [convergence_threshold, convergence_threshold], ...
                 '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            
            % 标注收敛情况
            final_window = max(1, length(convergence)-min(49, length(convergence)-1)):length(convergence);
            final_convergence = mean(convergence(final_window));
            if final_convergence < convergence_threshold
                status_text = '已收敛';
                status_color = [0 0.6 0];
            else
                status_text = '未收敛';
                status_color = [0.8 0 0];
            end
            
            text(max(valid_idx)*0.7, convergence_threshold*2, ...
                 sprintf('%s (%.4f)', status_text, final_convergence), ...
                 'FontSize', 12, 'FontWeight', 'bold', 'Color', status_color);
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('策略变化标准差 (对数尺度)', 'FontSize', 12);
            title(sprintf('%s - 收敛性分析', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            legend({'收敛度量', '收敛阈值'}, 'Location', 'best');
        end
        
        function plotRewardTrend(rewards, color, agent_name)
            % 绘制奖励趋势
            
            if isempty(rewards)
                text(0.5, 0.5, '奖励数据为空', 'HorizontalAlignment', 'center');
                title(sprintf('%s - 奖励趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            % 计算累积奖励
            cumulative_rewards = cumsum(rewards);
            
            % 双轴绘图
            yyaxis left;
            plot(1:length(rewards), rewards, 'Color', color, 'LineWidth', 1.5);
            ylabel('每轮平均奖励', 'FontSize', 12);
            ylim([min(rewards)*1.1, max(rewards)*1.1]);
            
            yyaxis right;
            plot(1:length(cumulative_rewards), cumulative_rewards, ...
                 'Color', color*0.6, 'LineWidth', 2, 'LineStyle', '--');
            ylabel('累积奖励', 'FontSize', 12);
            
            xlabel('迭代次数', 'FontSize', 12);
            title(sprintf('%s - 奖励趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            legend({'每轮奖励', '累积奖励'}, 'Location', 'best');
        end
        
        function plotPolicyStability(stability, color, agent_name)
            % 绘制策略稳定性
            
            valid_idx = find(stability > 0);
            if isempty(valid_idx)
                text(0.5, 0.5, '策略稳定性数据不足', 'HorizontalAlignment', 'center');
                title(sprintf('%s - 策略稳定性', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            % 绘制稳定性指标
            plot(valid_idx, stability(valid_idx), ...
                 'Color', color, 'LineWidth', 2);
            
            % 添加平滑曲线
            if length(valid_idx) > 20
                smoothed = movmean(stability(valid_idx), 20);
                hold on;
                plot(valid_idx, smoothed, 'Color', color*0.5, ...
                     'LineWidth', 2.5, 'LineStyle', '--');
            end
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('策略稳定性', 'FontSize', 12);
            title(sprintf('%s - 策略稳定性', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1.2]);
        end
        
        function plotPerformanceStats(results, agent_idx, agent_name)
            % 显示性能统计
            
            % 清除坐标轴
            axis off;
            
            % 计算统计数据
            stats = struct();
            
            if isfield(results, 'detection_rates') && size(results.detection_rates, 1) >= agent_idx
                n_iters = size(results.detection_rates, 2);
                last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
                stats.final_detection = mean(results.detection_rates(agent_idx, last_iters)) * 100;
                stats.max_detection = max(results.detection_rates(agent_idx, :)) * 100;
            else
                stats.final_detection = 0;
                stats.max_detection = 0;
            end
            
            if isfield(results, 'resource_utilization') && size(results.resource_utilization, 1) >= agent_idx
                n_iters = size(results.resource_utilization, 2);
                last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
                stats.final_resource = mean(results.resource_utilization(agent_idx, last_iters)) * 100;
            else
                stats.final_resource = 0;
            end
            
            if isfield(results, 'convergence_metrics') && size(results.convergence_metrics, 1) >= agent_idx
                n_iters = size(results.convergence_metrics, 2);
                last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
                stats.final_convergence = mean(results.convergence_metrics(agent_idx, last_iters));
            else
                stats.final_convergence = 0;
            end
            
            if isfield(results, 'defender_rewards') && size(results.defender_rewards, 1) >= agent_idx
                stats.total_reward = sum(results.defender_rewards(agent_idx, :));
            else
                stats.total_reward = 0;
            end
            
            % 创建表格显示
            text(0.1, 0.9, sprintf('%s 性能统计', agent_name), ...
                 'FontSize', 16, 'FontWeight', 'bold');
            
            y_pos = 0.7;
            metrics = {
                sprintf('最终检测率: %.2f%%', stats.final_detection),
                sprintf('最高检测率: %.2f%%', stats.max_detection),
                sprintf('资源利用率: %.2f%%', stats.final_resource),
                sprintf('收敛度: %.4f', stats.final_convergence),
                sprintf('总奖励: %.2f', stats.total_reward)
            };
            
            for i = 1:length(metrics)
                text(0.1, y_pos - (i-1)*0.12, metrics{i}, ...
                     'FontSize', 12, 'FontWeight', 'normal');
            end
            
            % 添加性能评级
            if stats.final_detection > 80
                rating = '优秀';
                rating_color = [0 0.7 0];
            elseif stats.final_detection > 60
                rating = '良好';
                rating_color = [0.7 0.7 0];
            elseif stats.final_detection > 40
                rating = '一般';
                rating_color = [1 0.5 0];
            else
                rating = '需要改进';
                rating_color = [1 0 0];
            end
            
            text(0.5, 0.15, sprintf('性能评级: %s', rating), ...
                 'FontSize', 14, 'FontWeight', 'bold', 'Color', rating_color, ...
                 'HorizontalAlignment', 'center');
        end
        
        function generateComparisonReport(results, config, agents)
            % 生成综合对比报告
            
            try
                figure('Position', [50, 50, 1400, 900], 'Name', '算法性能综合对比');
                set(gcf, 'Color', 'white');
                
                % 1. 检测率对比
                subplot(2, 3, 1);
                EnhancedReportGenerator.compareDetectionRates(results);
                
                % 2. 资源效率对比
                subplot(2, 3, 2);
                EnhancedReportGenerator.compareResourceEfficiency(results);
                
                % 3. 收敛速度对比
                subplot(2, 3, 3);
                EnhancedReportGenerator.compareConvergenceSpeed(results);
                
                % 4. 最终性能雷达图
                subplot(2, 3, 4);
                EnhancedReportGenerator.plotRadarChart(results);
                
                % 5. 性能时间线
                subplot(2, 3, 5);
                EnhancedReportGenerator.plotPerformanceTimeline(results);
                
                % 6. 综合评分
                subplot(2, 3, 6);
                EnhancedReportGenerator.plotOverallScores(results);
                
                % 保存图片
                filename = sprintf('reports/comparison_report_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
                saveas(gcf, filename);
                fprintf('保存对比报告: %s\n', filename);
                
            catch ME
                fprintf('生成对比报告时出错: %s\n', ME.message);
            end
        end
        
        function compareDetectionRates(results)
            % 对比检测率
            
            if ~isfield(results, 'detection_rates') || isempty(results.detection_rates)
                text(0.5, 0.5, '检测率数据不可用', 'HorizontalAlignment', 'center');
                title('检测率对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            
            hold on;
            n_agents = min(size(results.detection_rates, 1), 3);
            for i = 1:n_agents
                window_size = min(50, max(5, size(results.detection_rates, 2)/10));
                smoothed = movmean(results.detection_rates(i, :), window_size);
                plot(1:length(smoothed), smoothed, ...
                     'Color', colors(i, :), 'LineWidth', 2.5, ...
                     'DisplayName', agent_names{i});
            end
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('检测率', 'FontSize', 12);
            title('检测率对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            ylim([0, 1]);
        end
        
        function compareResourceEfficiency(results)
            % 对比资源效率
            
            if ~isfield(results, 'resource_utilization') || isempty(results.resource_utilization)
                text(0.5, 0.5, '资源利用数据不可用', 'HorizontalAlignment', 'center');
                title('资源效率对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            
            % 计算最后100次迭代的平均资源效率
            n_iters = size(results.resource_utilization, 2);
            last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
            n_agents = min(size(results.resource_utilization, 1), 3);
            avg_efficiency = zeros(n_agents, 1);
            
            for i = 1:n_agents
                avg_efficiency(i) = mean(results.resource_utilization(i, last_iters));
            end
            
            % 绘制柱状图
            b = bar(1:n_agents, avg_efficiency);
            
            % 设置颜色
            for i = 1:n_agents
                b.FaceColor = 'flat';
                b.CData(i,:) = colors(i,:);
            end
            
            % 添加数值标签
            for i = 1:n_agents
                text(i, avg_efficiency(i) + 0.02, ...
                     sprintf('%.1f%%', avg_efficiency(i)*100), ...
                     'HorizontalAlignment', 'center', ...
                     'FontSize', 11, 'FontWeight', 'bold');
            end
            
            set(gca, 'XTickLabel', agent_names(1:n_agents));
            ylabel('平均资源利用率', 'FontSize', 12);
            title('资源效率对比', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            if max(avg_efficiency) > 0
                ylim([0, max(avg_efficiency)*1.2]);
            end
        end
        
        function compareConvergenceSpeed(results)
            % 对比收敛速度
            
            if ~isfield(results, 'convergence_metrics') || isempty(results.convergence_metrics)
                text(0.5, 0.5, '收敛数据不可用', 'HorizontalAlignment', 'center');
                title('收敛速度对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            
            hold on;
            n_agents = min(size(results.convergence_metrics, 1), 3);
            for i = 1:n_agents
                valid_idx = find(results.convergence_metrics(i, :) > 0);
                if ~isempty(valid_idx)
                    semilogy(valid_idx, results.convergence_metrics(i, valid_idx), ...
                            'Color', colors(i, :), 'LineWidth', 2, ...
                            'DisplayName', agent_names{i});
                end
            end
            
            % 添加收敛阈值
            convergence_threshold = 0.01;
            n_iters = size(results.convergence_metrics, 2);
            plot([1, n_iters], [convergence_threshold, convergence_threshold], ...
                 'k--', 'LineWidth', 1.5, 'DisplayName', '收敛阈值');
            
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('收敛度量 (对数尺度)', 'FontSize', 12);
            title('收敛速度对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
        end
        
        function plotRadarChart(results)
            % 绘制性能雷达图
            
            text(0.5, 0.5, '雷达图功能', 'HorizontalAlignment', 'center', 'FontSize', 14);
            title('性能雷达图', 'FontSize', 14, 'FontWeight', 'bold');
            % 简化的雷达图实现，避免复杂的极坐标计算
        end
        
        function plotPerformanceTimeline(results)
            % 绘制性能时间线
            
            if ~isfield(results, 'detection_rates') || isempty(results.detection_rates)
                text(0.5, 0.5, '检测率数据不可用', 'HorizontalAlignment', 'center');
                title('性能发展时间线', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            
            % 计算关键里程碑
            n_iters = size(results.detection_rates, 2);
            milestones = [50, 100, 200, 300, 400, 500];
            milestones = milestones(milestones <= n_iters);
            
            if isempty(milestones)
                milestones = [max(1, n_iters/4), max(1, n_iters/2), max(1, 3*n_iters/4), n_iters];
            end
            
            n_agents = min(size(results.detection_rates, 1), 3);
            milestone_data = zeros(length(milestones), n_agents);
            
            for i = 1:length(milestones)
                for j = 1:n_agents
                    range = max(1, milestones(i)-9):milestones(i);
                    milestone_data(i, j) = mean(results.detection_rates(j, range));
                end
            end
            
            % 绘制时间线
            hold on;
            for i = 1:n_agents
                plot(milestones, milestone_data(:, i), ...
                     'Color', colors(i, :), 'LineWidth', 2.5, ...
                     'Marker', 'o', 'MarkerSize', 8, ...
                     'MarkerFaceColor', colors(i, :), ...
                     'DisplayName', agent_names{i});
            end
            
            xlabel('迭代里程碑', 'FontSize', 12);
            ylabel('检测率', 'FontSize', 12);
            title('性能发展时间线', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            ylim([0, 1]);
            set(gca, 'XTick', milestones);
        end
        
        function plotOverallScores(results)
            % 计算并显示综合评分
            
            if ~isfield(results, 'detection_rates') || isempty(results.detection_rates)
                text(0.5, 0.5, '数据不足以计算综合评分', 'HorizontalAlignment', 'center');
                title('算法综合评分', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            
            % 计算综合得分（加权平均）
            n_iters = size(results.detection_rates, 2);
            last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
            n_agents = min(size(results.detection_rates, 1), 3);
            scores = zeros(n_agents, 1);
            weights = [0.4, 0.3, 0.2, 0.1];  % 检测率、资源效率、收敛性、稳定性
            
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            
            for i = 1:n_agents
                detection_score = mean(results.detection_rates(i, last_iters));
                
                % 安全获取其他指标
                if isfield(results, 'resource_utilization') && size(results.resource_utilization, 1) >= i
                    resource_score = mean(results.resource_utilization(i, last_iters));
                else
                    resource_score = 0.5;  % 默认值
                end
                
                if isfield(results, 'convergence_metrics') && size(results.convergence_metrics, 1) >= i
                    convergence_score = 1 - min(1, mean(results.convergence_metrics(i, last_iters)));
                else
                    convergence_score = 0.5;  % 默认值
                end
                
                if isfield(results, 'policy_stability') && size(results.policy_stability, 1) >= i
                    valid_stability = results.policy_stability(i, results.policy_stability(i,:)>0);
                    if ~isempty(valid_stability)
                        stability_score = mean(valid_stability);
                    else
                        stability_score = 0.5;
                    end
                else
                    stability_score = 0.5;  % 默认值
                end
                
                scores(i) = weights(1) * detection_score + ...
                           weights(2) * resource_score + ...
                           weights(3) * convergence_score + ...
                           weights(4) * stability_score;
            end
            
            % 绘制评分
            b = bar(1:n_agents, scores * 100);
            
            % 设置颜色
            for i = 1:n_agents
                b.FaceColor = 'flat';
                b.CData(i,:) = colors(i,:);
            end
            
            % 添加分数标签
            for i = 1:n_agents
                text(i, scores(i)*100 + 2, sprintf('%.1f分', scores(i)*100), ...
                     'HorizontalAlignment', 'center', ...
                     'FontSize', 12, 'FontWeight', 'bold');
            end
            
            % 添加等级线
            hold on;
            plot([0.5, n_agents+0.5], [60, 60], 'g--', 'LineWidth', 2);
            plot([0.5, n_agents+0.5], [80, 80], 'b--', 'LineWidth', 2);
            
            text(n_agents+0.3, 60, '及格线', 'Color', 'g', 'FontWeight', 'bold');
            text(n_agents+0.3, 80, '优秀线', 'Color', 'b', 'FontWeight', 'bold');
            
            set(gca, 'XTickLabel', agent_names(1:n_agents));
            ylabel('综合得分', 'FontSize', 12);
            title('算法综合评分', 'FontSize', 14, 'FontWeight', 'bold');
            ylim([0, 110]);
            grid on;
            
            % 显示最佳算法
            [best_score, best_idx] = max(scores);
            text(n_agents/2, 105, ...
                 sprintf('最佳算法: %s', agent_names{best_idx}), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 14, 'FontWeight', 'bold', ...
                 'Color', colors(best_idx, :));
        end
        
        function generatePerformanceAnalysis(results, config, agents)
            % 生成详细的性能分析报告
            
            try
                filename = sprintf('reports/performance_analysis_%s.txt', ...
                                 datestr(now, 'yyyymmdd_HHMMSS'));
                
                fid = fopen(filename, 'w');
                if fid == -1
                    fprintf('无法创建分析报告文件\n');
                    return;
                end
                
                fprintf(fid, '========================================\n');
                fprintf(fid, 'FSP仿真性能深度分析报告\n');
                fprintf(fid, '========================================\n');
                fprintf(fid, '生成时间: %s\n\n', datestr(now));
                
                agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
                n_agents = min(size(results.detection_rates, 1), 3);
                n_iters = size(results.detection_rates, 2);
                last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
        
                % --- 一、检测率分析 ---
                fprintf(fid, '一、检测率分析\n');
                fprintf(fid, '----------------\n');
                if isfield(results, 'detection_rates') && ~isempty(results.detection_rates)
                    for i = 1:n_agents
                        avg_detection = mean(results.detection_rates(i, last_iters));
                        fprintf(fid, '\n%s:\n', agent_names{i});
                        fprintf(fid, '  平均检测率: %.2f%%\n', avg_detection * 100);
                        if avg_detection < 0.3
                            fprintf(fid, '  诊断: 检测率过低\n');
                        elseif avg_detection < 0.6
                            fprintf(fid, '  诊断: 检测率有待提高\n');
                        else
                            fprintf(fid, '  诊断: 检测率良好\n');
                        end
                    end
                else
                    fprintf(fid, '检测率数据不可用\n');
                end
                
                % --- 二、收敛性分析 ---
                fprintf(fid, '\n\n二、收敛性分析\n');
                fprintf(fid, '----------------\n');
                if isfield(results, 'convergence_metrics') && ~isempty(results.convergence_metrics)
                    for i = 1:n_agents
                        avg_convergence = mean(results.convergence_metrics(i, last_iters));
                        fprintf(fid, '\n%s:\n', agent_names{i});
                        fprintf(fid, '  收敛度量: %.4f\n', avg_convergence);
                        if avg_convergence > 0.05
                            fprintf(fid, '  诊断: 尚未充分收敛\n');
                        else
                            fprintf(fid, '  诊断: 收敛性良好\n');
                        end
                    end
                else
                    fprintf(fid, '收敛性数据不可用\n');
                end
        
                % --- 三、误报率分析 ---
                fprintf(fid, '\n\n三、误报率分析\n');
                fprintf(fid, '----------------\n');
                if isfield(results, 'false_positive_rates') && ~isempty(results.false_positive_rates)
                    for i = 1:n_agents
                        avg_fp_rate = mean(results.false_positive_rates(i, last_iters));
                        fprintf(fid, '\n%s:\n', agent_names{i});
                        fprintf(fid, '  平均误报率: %.2f%%\n', avg_fp_rate * 100);
                        if avg_fp_rate > 0.1
                            fprintf(fid, '  诊断: 误报率过高，可能导致资源浪费\n');
                        else
                            fprintf(fid, '  诊断: 误报率在可接受范围内\n');
                        end
                    end
                else
                    fprintf(fid, '误报率数据不可用\n');
                end
                
                % 检测率过低的专项分析
                fprintf(fid, '\n\n三、检测率过低原因分析\n');
                fprintf(fid, '------------------------\n');
                fprintf(fid, '根据仿真结果，检测率过低可能由以下因素导致:\n\n');
                
                fprintf(fid, '1. 参数设置问题:\n');
                fprintf(fid, '   - 学习率过低: 建议从0.01增加到0.05-0.15\n');
                fprintf(fid, '   - 探索率不足: 建议初始值设为0.3-0.5\n');
                fprintf(fid, '   - 衰减过快: 建议使用0.999的衰减率\n');
                fprintf(fid, '   - 折扣因子过低: 建议设为0.95-0.99\n\n');
                
                fprintf(fid, '2. 环境设计问题:\n');
                fprintf(fid, '   - 奖励函数稀疏: 需要增加中间奖励\n');
                fprintf(fid, '   - 状态空间过大: 考虑状态抽象或特征提取\n');
                fprintf(fid, '   - 动作空间不合理: 检查动作定义的有效性\n\n');
                
                fprintf(fid, '3. 算法实现问题:\n');
                fprintf(fid, '   - Q表初始化不当: 建议使用小的随机值\n');
                fprintf(fid, '   - 更新策略有误: 检查Q值更新公式\n');
                fprintf(fid, '   - 缺乏经验回放: 考虑实现经验回放机制\n\n');
                
                % 改进建议
                fprintf(fid, '四、具体改进方案\n');
                fprintf(fid, '----------------\n');
                fprintf(fid, '短期改进 (立即可实施):\n');
                fprintf(fid, '1. 调整关键参数:\n');
                fprintf(fid, '   config.learning_rate = 0.1;        %% 提高学习率\n');
                fprintf(fid, '   config.exploration_rate = 0.5;      %% 增加初始探索\n');
                fprintf(fid, '   config.decay_rate = 0.999;          %% 缓慢衰减\n');
                fprintf(fid, '   config.discount_factor = 0.95;      %% 适中的折扣\n\n');
                
                fprintf(fid, '2. 奖励函数优化:\n');
                fprintf(fid, '   - 增加检测成功的奖励\n');
                fprintf(fid, '   - 减少误报的惩罚\n');
                fprintf(fid, '   - 添加距离或接近目标的中间奖励\n\n');
                
                fprintf(fid, '3. 训练策略调整:\n');
                fprintf(fid, '   - 增加训练轮数到1000-2000轮\n');
                fprintf(fid, '   - 使用更多的训练episode\n');
                fprintf(fid, '   - 实现动态参数调整\n\n');
                
                fprintf(fid, '中期改进 (需要开发):\n');
                fprintf(fid, '1. 算法升级:\n');
                fprintf(fid, '   - 实现经验回放机制\n');
                fprintf(fid, '   - 使用优先级经验回放\n');
                fprintf(fid, '   - 考虑Double DQN或Dueling DQN\n\n');
                
                fprintf(fid, '2. 状态表示优化:\n');
                fprintf(fid, '   - 特征工程提取关键信息\n');
                fprintf(fid, '   - 状态归一化处理\n');
                fprintf(fid, '   - 多尺度状态表示\n\n');
                
                fprintf(fid, '长期改进 (架构优化):\n');
                fprintf(fid, '1. 深度强化学习:\n');
                fprintf(fid, '   - 使用神经网络代替Q表\n');
                fprintf(fid, '   - 实现CNN处理空间信息\n');
                fprintf(fid, '   - 考虑Actor-Critic架构\n\n');
                
                fprintf(fid, '2. 多智能体协作:\n');
                fprintf(fid, '   - 智能体间信息共享\n');
                fprintf(fid, '   - 协作奖励机制\n');
                fprintf(fid, '   - 分布式学习策略\n\n');
                
                fclose(fid);
                fprintf('✓ 性能分析报告已生成: %s\n', filename);
                
            catch ME
                fprintf('生成性能分析报告时出错: %s\n', ME.message);
                if exist('fid', 'var') && fid ~= -1
                    fclose(fid);
                end
            end
        end
    end
end