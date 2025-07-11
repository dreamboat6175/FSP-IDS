%% EnhancedReportGenerator.m - 增强版报告生成器（RADI版）
classdef EnhancedReportGenerator < handle
    methods (Static)
        function generateEnhancedReport(results, config, agents, ~)
            % 生成增强的分析报告（基于RADI体系）
            try
                if ~exist('reports', 'dir')
                    mkdir('reports');
                end
                fprintf('正在生成RADI增强报告...\n');
                EnhancedReportGenerator.generateIndividualReports(results, config, agents);
                EnhancedReportGenerator.generateComparisonReport(results, config, agents);
                EnhancedReportGenerator.generatePerformanceAnalysis(results, config, agents);
                fprintf('✓ RADI增强报告生成完成！\n');
            catch ME
                fprintf('报告生成出错: %s\n', ME.message);
                fprintf('错误详情: %s\n', getReport(ME, 'extended', 'hyperlinks', 'on'));
                fprintf('继续运行简化报告生成...\n');
                EnhancedReportGenerator.generateSimpleReport(results, config);
            end
        end
        function generateSimpleReport(results, config)
            % 生成简化报告（RADI版）
            fprintf('\n=== FSP仿真简化报告（RADI） ===\n');
            fprintf('生成时间: %s\n', datestr(now));
            if isfield(results, 'radi') && ~isempty(results.radi)
                agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
                for i = 1:min(size(results.radi, 1), 3)
                    last_100 = max(1, size(results.radi, 2)-99):size(results.radi, 2);
                    avg_radi = mean(results.radi(i, last_100));
                    fprintf('\n%s:\n', agent_names{i});
                    fprintf('  最终RADI: %.4f\n', avg_radi);
                    if avg_radi < config.radi.threshold_excellent
                        fprintf('  状态: RADI优秀，资源分配极优\n');
                    elseif avg_radi < config.radi.threshold_good
                        fprintf('  状态: RADI良好，有进一步优化空间\n');
                    elseif avg_radi < config.radi.threshold_acceptable
                        fprintf('  状态: RADI一般，建议优化分配\n');
                    else
                        fprintf('  状态: RADI偏高，需重点优化\n');
                    end
                end
            end
            save('reports/simple_results.mat', 'results', 'config');
            fprintf('\n结果已保存到 reports/simple_results.mat\n');
        end
        function generateIndividualReports(results, config, agents)
            % 为每个算法生成单独的RADI报告
            if ~isfield(results, 'n_agents') || results.n_agents == 0
                results.n_agents = min(3, size(results.radi, 1));
            end
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            for i = 1:results.n_agents
                try
                    figure('Position', [100, 100, 1200, 900], 'Name', sprintf('%s RADI性能分析', agent_names{i}), 'Visible', 'off');
                    set(gcf, 'Color', 'white');
                    % 1. RADI趋势
                    subplot(2, 2, 1);
                    EnhancedReportGenerator.plotRADITrend(results.radi(i, :), colors(i, :), agent_names{i}, config);
                    % 2. 资源利用效率
                    subplot(2, 2, 2);
                    if isfield(results, 'resource_efficiency') && size(results.resource_efficiency, 1) >= i
                        EnhancedReportGenerator.plotResourceEfficiency(results.resource_efficiency(i, :), colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '资源效率数据不可用', 'HorizontalAlignment', 'center');
                    end
                    % 3. 分配均衡性
                    subplot(2, 2, 3);
                    if isfield(results, 'allocation_balance') && size(results.allocation_balance, 1) >= i
                        EnhancedReportGenerator.plotAllocationBalance(results.allocation_balance(i, :), colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '均衡性数据不可用', 'HorizontalAlignment', 'center');
                    end
                    % 4. 收敛性分析
                    subplot(2, 2, 4);
                    if isfield(results, 'convergence_metrics') && size(results.convergence_metrics, 1) >= i
                        EnhancedReportGenerator.plotConvergenceAnalysis(results.convergence_metrics(i, :), colors(i, :), agent_names{i});
                    else
                        text(0.5, 0.5, '收敛性数据不可用', 'HorizontalAlignment', 'center');
                    end
                    % 保存图片
                    filename = sprintf('reports/%s_detailed_report_%s.png', strrep(agent_names{i}, '-', '_'), datestr(now, 'yyyymmdd_HHMMSS'));
                    saveas(gcf, filename);
                    fprintf('保存个体RADI报告: %s\n', filename);
                    close(gcf);
                catch ME
                    fprintf('生成 %s 个体RADI报告时出错: %s\n', agent_names{i}, ME.message);
                end
            end
        end
        function plotRADITrend(radi, color, agent_name, config)
            % 绘制RADI趋势（越低越好）
            if isempty(radi) || all(radi == 0)
                text(0.5, 0.5, 'RADI数据为空', 'HorizontalAlignment', 'center');
                return;
            end
            plot(1:length(radi), radi, 'Color', [color 0.3], 'LineWidth', 0.5);
            hold on;
            window_size = min(50, max(5, floor(length(radi)/10)));
            if length(radi) >= window_size
                smoothed = movmean(radi, window_size);
                plot(1:length(smoothed), smoothed, 'Color', color, 'LineWidth', 2.5);
            end
            if length(radi) > 10
                x = 1:length(radi);
                p = polyfit(x, radi, 1);
                trend = polyval(p, x);
                plot(x, trend, '--', 'Color', color*0.5, 'LineWidth', 2);
            end
            final_window = max(1, length(radi)-min(99, length(radi)-1)):length(radi);
            final_radi = mean(radi(final_window));
            text(length(radi)*0.7, min(radi)+0.05, sprintf('最终RADI: %.4f', final_radi), 'FontSize', 12, 'FontWeight', 'bold', 'Color', color);
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('RADI', 'FontSize', 12);
            title(sprintf('%s - RADI趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, max(0.5, max(radi)*1.1)]);
            % 阈值线
            yline(config.radi.threshold_excellent, ':', '优秀', 'Color', [0 0.6 0], 'LabelHorizontalAlignment', 'left');
            yline(config.radi.threshold_good, ':', '良好', 'Color', [0.7 0.7 0], 'LabelHorizontalAlignment', 'left');
            yline(config.radi.threshold_acceptable, ':', '可接受', 'Color', [1 0.5 0], 'LabelHorizontalAlignment', 'left');
            legend({'原始数据', '平滑曲线', '趋势线'}, 'Location', 'best');
        end
        function plotResourceEfficiency(resource_eff, color, agent_name)
            % 绘制资源利用效率
            if isempty(resource_eff)
                text(0.5, 0.5, '资源效率数据为空', 'HorizontalAlignment', 'center');
                return;
            end
            x = 1:length(resource_eff);
            fill([x, fliplr(x)], [resource_eff, zeros(size(resource_eff))], color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            hold on;
            plot(x, resource_eff, 'Color', color, 'LineWidth', 2);
            avg_eff = mean(resource_eff);
            plot([1, length(resource_eff)], [avg_eff, avg_eff], '--', 'Color', color*0.5, 'LineWidth', 2);
            text(length(resource_eff)*0.5, avg_eff+0.05, sprintf('平均: %.1f%%', avg_eff*100), 'FontSize', 11, 'HorizontalAlignment', 'center');
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('资源利用率', 'FontSize', 12);
            title(sprintf('%s - 资源利用效率', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            ylim([0, 1]);
        end
        function plotAllocationBalance(balance, color, agent_name)
            % 绘制分配均衡性趋势（越高越均衡）
            if isempty(balance)
                text(0.5, 0.5, '均衡性数据为空', 'HorizontalAlignment', 'center');
                return;
            end
            x = 1:length(balance);
            plot(x, balance, 'Color', color, 'LineWidth', 2);
            hold on;
            avg_balance = mean(balance);
            plot([1, length(balance)], [avg_balance, avg_balance], '--', 'Color', color*0.5, 'LineWidth', 2);
            text(length(balance)*0.5, avg_balance+0.05, sprintf('平均: %.2f', avg_balance), 'FontSize', 11, 'HorizontalAlignment', 'center');
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('分配均衡性', 'FontSize', 12);
            title(sprintf('%s - 分配均衡性趋势', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
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
            semilogy(valid_idx, convergence(valid_idx), 'Color', color, 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4, 'MarkerFaceColor', color);
            convergence_threshold = 0.01;
            hold on;
            plot([min(valid_idx), max(valid_idx)], [convergence_threshold, convergence_threshold], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
            final_window = max(1, length(convergence)-min(49, length(convergence)-1)):length(convergence);
            final_convergence = mean(convergence(final_window));
            if final_convergence < convergence_threshold
                status_text = '已收敛'; status_color = [0 0.6 0];
            else
                status_text = '未收敛'; status_color = [0.8 0 0];
            end
            text(max(valid_idx)*0.7, convergence_threshold*2, sprintf('%s (%.4f)', status_text, final_convergence), 'FontSize', 12, 'FontWeight', 'bold', 'Color', status_color);
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('策略变化标准差 (对数尺度)', 'FontSize', 12);
            title(sprintf('%s - 收敛性分析', agent_name), 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            legend({'收敛度量', '收敛阈值'}, 'Location', 'best');
        end
        function generateComparisonReport(results, config, agents)
            % 生成综合对比报告（RADI体系）
            try
                figure('Position', [50, 50, 1400, 900], 'Name', '算法RADI性能综合对比', 'Visible', 'off');
                set(gcf, 'Color', 'white');
                % 1. RADI对比
                subplot(2, 2, 1);
                EnhancedReportGenerator.compareRADI(results, config);
                % 2. 资源效率对比
                subplot(2, 2, 2);
                EnhancedReportGenerator.compareResourceEfficiency(results);
                % 3. 分配均衡性对比
                subplot(2, 2, 3);
                EnhancedReportGenerator.compareAllocationBalance(results);
                % 4. 收敛速度对比
                subplot(2, 2, 4);
                EnhancedReportGenerator.compareConvergenceSpeed(results);
                filename = sprintf('reports/comparison_report_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
                saveas(gcf, filename);
                fprintf('保存RADI对比报告: %s\n', filename);
                close(gcf);
            catch ME
                fprintf('生成RADI对比报告时出错: %s\n', ME.message);
            end
        end
        function compareRADI(results, config)
            % 对比RADI
            if ~isfield(results, 'radi') || isempty(results.radi)
                text(0.5, 0.5, 'RADI数据不可用', 'HorizontalAlignment', 'center');
                title('RADI对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            hold on;
            n_agents = min(size(results.radi, 1), 3);
            for i = 1:n_agents
                window_size = min(50, max(5, floor(size(results.radi, 2)/10)));
                smoothed = movmean(results.radi(i, :), window_size);
                plot(1:length(smoothed), smoothed, 'Color', colors(i, :), 'LineWidth', 2.5, 'DisplayName', agent_names{i});
            end
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('RADI', 'FontSize', 12);
            title('RADI对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
            ylim([0, max(0.5, max(results.radi(:))*1.1)]);
            yline(config.radi.threshold_excellent, ':', '优秀', 'Color', [0 0.6 0], 'LabelHorizontalAlignment', 'left');
            yline(config.radi.threshold_good, ':', '良好', 'Color', [0.7 0.7 0], 'LabelHorizontalAlignment', 'left');
            yline(config.radi.threshold_acceptable, ':', '可接受', 'Color', [1 0.5 0], 'LabelHorizontalAlignment', 'left');
        end
        function compareResourceEfficiency(results)
            % 对比资源效率
            if ~isfield(results, 'resource_efficiency') || isempty(results.resource_efficiency)
                text(0.5, 0.5, '资源效率数据不可用', 'HorizontalAlignment', 'center');
                title('资源效率对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            n_iters = size(results.resource_efficiency, 2);
            last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
            n_agents = min(size(results.resource_efficiency, 1), 3);
            avg_efficiency = zeros(n_agents, 1);
            for i = 1:n_agents
                avg_efficiency(i) = mean(results.resource_efficiency(i, last_iters));
            end
            b = bar(1:n_agents, avg_efficiency);
            for i = 1:n_agents
                b.FaceColor = 'flat';
                b.CData(i,:) = colors(i,:);
            end
            for i = 1:n_agents
                text(i, avg_efficiency(i) + 0.02, sprintf('%.1f%%', avg_efficiency(i)*100), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
            end
            set(gca, 'XTickLabel', agent_names(1:n_agents));
            ylabel('平均资源利用率', 'FontSize', 12);
            title('资源效率对比', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            if max(avg_efficiency) > 0
                ylim([0, max(avg_efficiency)*1.2]);
            end
        end
        function compareAllocationBalance(results)
            % 对比分配均衡性
            if ~isfield(results, 'allocation_balance') || isempty(results.allocation_balance)
                text(0.5, 0.5, '均衡性数据不可用', 'HorizontalAlignment', 'center');
                title('分配均衡性对比', 'FontSize', 14, 'FontWeight', 'bold');
                return;
            end
            colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.2 0.6 0.2];
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            n_iters = size(results.allocation_balance, 2);
            last_iters = max(1, n_iters-min(99, n_iters-1)):n_iters;
            n_agents = min(size(results.allocation_balance, 1), 3);
            avg_balance = zeros(n_agents, 1);
            for i = 1:n_agents
                avg_balance(i) = mean(results.allocation_balance(i, last_iters));
            end
            b = bar(1:n_agents, avg_balance);
            for i = 1:n_agents
                b.FaceColor = 'flat';
                b.CData(i,:) = colors(i,:);
            end
            for i = 1:n_agents
                text(i, avg_balance(i) + 0.02, sprintf('%.2f', avg_balance(i)), 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
            end
            set(gca, 'XTickLabel', agent_names(1:n_agents));
            ylabel('平均分配均衡性', 'FontSize', 12);
            title('分配均衡性对比', 'FontSize', 14, 'FontWeight', 'bold');
            grid on;
            if max(avg_balance) > 0
                ylim([0, max(avg_balance)*1.2]);
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
                    semilogy(valid_idx, results.convergence_metrics(i, valid_idx), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', agent_names{i});
                end
            end
            convergence_threshold = 0.01;
            n_iters = size(results.convergence_metrics, 2);
            plot([1, n_iters], [convergence_threshold, convergence_threshold], 'k--', 'LineWidth', 1.5, 'DisplayName', '收敛阈值');
            xlabel('迭代次数', 'FontSize', 12);
            ylabel('收敛度量 (对数尺度)', 'FontSize', 12);
            title('收敛速度对比', 'FontSize', 14, 'FontWeight', 'bold');
            legend('Location', 'best');
            grid on;
        end
        function generatePerformanceAnalysis(results, config, agents)
            % 核心性能分析报告（RADI体系）
            try
                fprintf('正在生成RADI性能分析报告...\n');
                EnhancedReportGenerator.generateOptimizationReport(results, config);
                fprintf('✓ RADI性能分析报告生成完成！\n');
            catch ME
                fprintf('RADI性能分析报告生成出错: %s\n', ME.message);
            end
        end
        function generateOptimizationReport(results, config)
            % 生成RADI优化效果报告
            filename = sprintf('reports/optimization_report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));
            fid = fopen(filename, 'w');
            fprintf(fid, '========================================\n');
            fprintf(fid, 'FSP-TCS RADI优化效果分析报告\n');
            fprintf(fid, '========================================\n');
            fprintf(fid, '生成时间: %s\n\n', datestr(now));
            fprintf(fid, '一、RADI与资源分配优化配置\n');
            fprintf(fid, '----------------\n');
            fprintf(fid, '1. 资源分配目标: %s\n', mat2str(config.radi.optimal_allocation));
            fprintf(fid, '2. RADI权重: computation=%.2f, bandwidth=%.2f, sensors=%.2f, scanning=%.2f, inspection=%.2f\n', ...
                config.radi.weight_computation, config.radi.weight_bandwidth, config.radi.weight_sensors, config.radi.weight_scanning, config.radi.weight_inspection);
            fprintf(fid, '3. RADI阈值: 优秀<%.2f, 良好<%.2f, 可接受<%.2f\n', config.radi.threshold_excellent, config.radi.threshold_good, config.radi.threshold_acceptable);
            fprintf(fid, '\n二、性能指标对比（RADI为主）\n');
            fprintf(fid, '----------------\n');
            last_iters = max(1, results.n_iterations-99):results.n_iterations;
            agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            fprintf(fid, '算法\t\tRADI\t资源效率\t均衡性\n');
            fprintf(fid, '----\t\t------\t------\t------\n');
            for i = 1:results.n_agents
                radi = mean(results.radi(i, last_iters));
                eff = mean(results.resource_efficiency(i, last_iters));
                bal = mean(results.allocation_balance(i, last_iters));
                fprintf(fid, '%s\t%.4f\t%.2f%%\t%.2f\n', agent_names{i}, radi, eff*100, bal);
            end
            fprintf(fid, '\n三、进一步优化建议（基于RADI）\n');
            fprintf(fid, '--------------------\n');
            [~, worst_idx] = max(mean(results.radi(:, last_iters), 2));
            fprintf(fid, '\n%s 算法RADI最高，建议:\n', agent_names{worst_idx});
            if mean(results.radi(worst_idx, last_iters)) > config.radi.threshold_acceptable
                fprintf(fid, '• RADI偏高 - 建议优化资源分配策略，提升分配均衡性和效率。\n');
            elseif mean(results.radi(worst_idx, last_iters)) > config.radi.threshold_good
                fprintf(fid, '• RADI一般 - 可进一步微调分配权重或增加训练轮数。\n');
            else
                fprintf(fid, '• RADI表现良好。\n');
            end
            fprintf(fid, '\n四、系统部署建议\n');
            fprintf(fid, '----------------\n');
            fprintf(fid, '1. 选择RADI最低的算法进行部署\n');
            fprintf(fid, '2. 持续监控RADI和资源效率，动态调整分配策略\n');
            fprintf(fid, '3. 定期复训以适应新威胁和环境变化\n');
            fclose(fid);
            fprintf('RADI优化报告已生成: %s\n', filename);
        end
    end
end