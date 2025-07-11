%% EnhancedReportGenerator.m - 优化版报告生成器

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

                % 为每个算法生成单独的详细报告
                EnhancedReportGenerator.generateIndividualReports(results, config, agents);

                % 生成综合对比报告
                EnhancedReportGenerator.generateComparisonReport(results, config, agents);

                % 生成性能分析报告
                EnhancedReportGenerator.generatePerformanceAnalysis(results, config, agents);

                fprintf('✓ 增强报告生成完成！\n');
            catch ME
                fprintf('报告生成出错: %s\n', ME.message);
                fprintf('错误详情: %s\n', getReport(ME, 'extended', 'hyperlinks', 'on'));
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
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2]; % 蓝色系，红色系，绿色系

            for i = 1:results.n_agents
                try
                    % 生成个体性能报告
                    figure('Position', [100, 100, 1200, 900], 'Name', sprintf('%s 性能分析', agent_names{i}), 'Visible', 'off');
                    set(gcf, 'Color', 'white');

                    % 创建子图，分别显示检测率、误报率和收敛性
                    EnhancedReportGenerator.createSubplots(results, i, colors(i, :), agent_names{i});

                    % 保存图像
                    EnhancedReportGenerator.saveFigure(agent_names{i});
                catch ME
                    fprintf('生成 %s 个体报告时出错: %s\n', agent_names{i}, ME.message);
                end
            end
        end

        function createSubplots(results, i, color, agent_name)
            % 创建子图，显示检测率、误报率、收敛性等指标

            subplot(2, 2, 1); % 检测率趋势
            EnhancedReportGenerator.plotDetectionTrend(results.detection_rates(i, :), color, agent_name);

            subplot(2, 2, 2); % 误报率趋势
            EnhancedReportGenerator.plotFalsePositiveRate(results.false_positive_rates(i, :), color, agent_name);

            subplot(2, 2, 3); % 收敛性分析
            EnhancedReportGenerator.plotConvergenceAnalysis(results.convergence_metrics(i, :), color, agent_name);

            subplot(2, 2, 4); % 性能统计
            EnhancedReportGenerator.plotPerformanceStats(results, i, agent_name);
        end

        function saveFigure(agent_name)
            % 保存图形
            filename = sprintf('reports/%s_detailed_report_%s.png', strrep(agent_name, '-', '_'), datestr(now, 'yyyymmdd_HHMMSS'));
            saveas(gcf, filename);
            fprintf('保存个体报告: %s\n', filename);
            close(gcf);
        end

        function plotDetectionTrend(detection_rates, color, agent_name)
            % 绘制检测率趋势（带平滑和置信区间）

            if isempty(detection_rates) || all(detection_rates == 0)
                text(0.5, 0.5, '检测率数据为空', 'HorizontalAlignment', 'center');
                return;
            end

            % 绘制原始数据，平滑曲线和趋势线
            plot(1:length(detection_rates), detection_rates, 'Color', [color 0.3], 'LineWidth', 0.5);
            hold on;
            smoothed = movmean(detection_rates, min(50, max(5, floor(length(detection_rates)/10))));
            plot(1:length(smoothed), smoothed, 'Color', color, 'LineWidth', 2.5);
            p = polyfit(1:length(detection_rates), detection_rates, 1);
            trend = polyval(p, 1:length(detection_rates));
            plot(1:length(detection_rates), trend, '--', 'Color', color*0.5, 'LineWidth', 2);

            % 显示最终检测率
            final_rate = mean(detection_rates(max(1, length(detection_rates)-99):end));
            text(length(detection_rates)*0.7, max(0.1, max(detection_rates)*0.9), sprintf('最终检测率: %.1f%%', final_rate*100), 'FontSize', 12, 'FontWeight', 'bold', 'Color', color);

            xlabel('迭代次数', 'FontSize', 12);
            ylabel('检测率', 'FontSize', 12);
            title(sprintf('%s - 检测率趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1]);
            legend({'原始数据', '平滑曲线', '趋势线'}, 'Location', 'best');
        end

        function plotFalsePositiveRate(fp_rates, color, agent_name)
            % 绘制误报率趋势（带平滑和置信区间）

            if isempty(fp_rates)
                text(0.5, 0.5, '误报率数据为空', 'HorizontalAlignment', 'center');
                title(sprintf('%s - 误报率趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end

            % 绘制误报率数据和移动平均曲线
            plot(1:length(fp_rates), fp_rates, 'Color', [color 0.3], 'LineWidth', 0.5);
            hold on;
            smoothed = movmean(fp_rates, min(50, max(5, floor(length(fp_rates)/10))));
            plot(1:length(smoothed), smoothed, 'Color', color, 'LineWidth', 2.5);

            final_fp_rate = mean(fp_rates(max(1, end-99):end));
            text(length(fp_rates)*0.7, max(0.1, max(fp_rates)*0.9), sprintf('最终误报率: %.2f%%', final_fp_rate*100), 'FontSize', 12, 'FontWeight', 'bold', 'Color', color);

            xlabel('迭代次数', 'FontSize', 12);
            ylabel('误报率 (FPR)', 'FontSize', 12);
            title(sprintf('%s - 误报率趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 0.5]); % 误报率通常较低，调整Y轴范围以便观察
            legend({'原始数据', '平滑曲线'}, 'Location', 'best');
        end

        function plotPerformanceStats(results, agent_idx, agent_name)
            % 显示性能统计

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
            text(0.1, 0.9, sprintf('%s 性能统计', agent_name), 'FontSize', 16, 'FontWeight', 'bold');

            y_pos = 0.7;
            metrics = {
                sprintf('最终检测率: %.2f%%', stats.final_detection),
                sprintf('最高检测率: %.2f%%', stats.max_detection),
                sprintf('资源利用率: %.2f%%', stats.final_resource),
                sprintf('收敛度: %.4f', stats.final_convergence),
                sprintf('总奖励: %.2f', stats.total_reward)
            };

            for i = 1:length(metrics)
                text(0.1, y_pos - (i-1)*0.12, metrics{i}, 'FontSize', 12, 'FontWeight', 'normal');
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

            text(0.5, 0.15, sprintf('性能评级: %s', rating), 'FontSize', 14, 'FontWeight', 'bold', 'Color', rating_color, 'HorizontalAlignment', 'center');
        end

        function generateComparisonReport(results, config, agents)
            % 生成综合对比报告

            try
                figure('Position', [50, 50, 1400, 900], 'Name', '算法性能综合对比', 'Visible', 'off');
                set(gcf, 'Color', 'white');

                % 1. 检测率对比
                subplot(2, 2, 1);
                EnhancedReportGenerator.compareDetectionRates(results);

                % 2. 误报率对比
                subplot(2, 2, 2);
                EnhancedReportGenerator.compareFalsePositiveRates(results);

                % 3. 资源效率对比
                subplot(2, 2, 3);
                EnhancedReportGenerator.compareResourceEfficiency(results);

                % 4. 收敛速度对比
                subplot(2, 2, 4);
                EnhancedReportGenerator.compareConvergenceSpeed(results);

                % 保存图表
                filename = sprintf('reports/comparison_report_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
                saveas(gcf, filename);
                fprintf('保存对比报告: %s\n', filename);
                close(gcf);
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
                window_size = min(50, max(5, floor(size(results.detection_rates, 2)/10)));
                smoothed = movmean(results.detection_rates(i, :), window_size);
                plot(1:length(smoothed), smoothed, 'Color', colors(i, :), 'LineWidth', 2.5, 'DisplayName', agent_names{i});
            end

            xlabel('迭代次数', 'FontSize', 12);
            ylabel('检测率', 'FontSize', 12);
            title('检测率对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            ylim([0, 1]);
        end

        function compareFalsePositiveRates(results)
            % 对比误报率

            if ~isfield(results, 'false_positive_rates') || isempty(results.false_positive_rates)
                text(0.5, 0.5, '误报率数据不可用', 'HorizontalAlignment', 'center');
                title('误报率对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end

            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};

            hold on;
            n_agents = min(size(results.false_positive_rates, 1), 3);
            for i = 1:n_agents
                window_size = min(50, max(5, floor(size(results.false_positive_rates, 2)/10)));
                smoothed = movmean(results.false_positive_rates(i, :), window_size);
                plot(1:length(smoothed), smoothed, 'Color', colors(i, :), 'LineWidth', 2.5, 'DisplayName', agent_names{i});
            end

            xlabel('迭代次数', 'FontSize', 12);
            ylabel('误报率 (FPR)', 'FontSize', 12);
            title('误报率对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            ylim([0, 0.5]);
        end
    end
end
