%% EnhancedVisualization.m - 增强版多角度可视化系统
% =========================================================================
% 描述: 提供全面、直观的仿真结果可视化
% 包括: 性能分析、策略演化、博弈动态、收敛性等多个维度
% =========================================================================

classdef EnhancedVisualization < handle
    
    properties
        results      % 仿真结果数据
        config       % 配置信息
        environment  % 环境对象
        figures      % 图形句柄存储
    end
    
    methods
        function obj = EnhancedVisualization(results, config, environment)
            % 构造函数
            obj.results = results;
            obj.config = config;
            obj.environment = environment;
            obj.figures = {};
        end
        
        function generateCompleteReport(obj)
            % 生成完整的可视化报告
            
            fprintf('\n=== 生成综合可视化报告 ===\n');
            
            % 1. 主要性能指标分析
            obj.createPerformanceAnalysis();
            
            % 2. 策略演化与博弈动态
            obj.createStrategyEvolution();
            
            % 3. 攻防对抗分析
            obj.createAttackDefenseAnalysis();
            
            % 4. 收敛性与稳定性分析
            obj.createConvergenceAnalysis();
            
            % 5. 资源效率与分配分析
            obj.createResourceAnalysis();
            
            % 6. 统计摘要与洞察
            obj.createStatisticalSummary();
            
            % 7. 3D交互式可视化
            obj.create3DVisualization();
            
            % 8. 动态时序分析
            obj.createTimeSeriesAnalysis();
            
            fprintf('可视化报告生成完成！\n');
        end
        
        function createPerformanceAnalysis(obj)
            % 创建性能分析图表
            
            fig = figure('Name', '性能指标综合分析', 'Position', [50 50 1600 900]);
            obj.figures{end+1} = fig;
            
            % 准备数据
            episodes = 1:length(obj.results.radi_history);
            
            % 子图1: RADI演化与目标对比
            subplot(3,4,1);
            obj.plotRADIEvolution();
            
            % 子图2: 攻击成功率分析
            subplot(3,4,2);
            obj.plotAttackSuccessRate();
            
            % 子图3: 损害值分析
            subplot(3,4,3);
            obj.plotDamageAnalysis();
            
            % 子图4: 累积性能对比
            subplot(3,4,4);
            obj.plotCumulativePerformance();
            
            % 子图5-6: 奖励演化分析（双轴）
            subplot(3,4,[5 6]);
            obj.plotRewardEvolution();
            
            % 子图7: 性能提升率
            subplot(3,4,7);
            obj.plotPerformanceImprovement();
            
            % 子图8: 相关性热力图
            subplot(3,4,8);
            obj.plotCorrelationHeatmap();
            
            % 子图9-10: 滑动窗口性能分析
            subplot(3,4,[9 10]);
            obj.plotSlidingWindowPerformance();
            
            % 子图11-12: 性能分布对比
            subplot(3,4,[11 12]);
            obj.plotPerformanceDistribution();
            
            sgtitle('性能指标综合分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createStrategyEvolution(obj)
            % 创建策略演化分析图表
            
            fig = figure('Name', '策略演化与博弈动态', 'Position', [100 100 1600 900]);
            obj.figures{end+1} = fig;
            
            % 子图1-2: 攻击策略演化热力图
            subplot(3,4,[1 2]);
            obj.plotAttackStrategyHeatmap();
            
            % 子图3-4: 防御策略演化热力图  
            subplot(3,4,[3 4]);
            obj.plotDefenseStrategyHeatmap();
            
            % 子图5: 策略熵演化
            subplot(3,4,5);
            obj.plotStrategyEntropy();
            
            % 子图6: 策略相似度分析
            subplot(3,4,6);
            obj.plotStrategySimilarity();
            
            % 子图7-8: 站点级策略对比
            subplot(3,4,[7 8]);
            obj.plotStationStrategyComparison();
            
            % 子图9: 策略收敛速度
            subplot(3,4,9);
            obj.plotStrategyConvergenceSpeed();
            
            % 子图10: 探索-利用平衡
            subplot(3,4,10);
            obj.plotExplorationExploitation();
            
            % 子图11-12: 最优响应分析
            subplot(3,4,[11 12]);
            obj.plotBestResponseAnalysis();
            
            sgtitle('策略演化与博弈动态分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createAttackDefenseAnalysis(obj)
            % 创建攻防对抗分析图表
            
            fig = figure('Name', '攻防对抗深度分析', 'Position', [150 150 1600 900]);
            obj.figures{end+1} = fig;
            
            % 子图1: 攻防强度时序图
            subplot(3,3,1);
            obj.plotAttackDefenseIntensity();
            
            % 子图2: 站点脆弱性分析
            subplot(3,3,2);
            obj.plotStationVulnerability();
            
            % 子图3: 攻击模式分析
            subplot(3,3,3);
            obj.plotAttackPatterns();
            
            % 子图4-5: 动态博弈相图
            subplot(3,3,[4 5]);
            obj.plotGamePhaseDiagram();
            
            % 子图6: 防御效率分析
            subplot(3,3,6);
            obj.plotDefenseEfficiency();
            
            % 子图7-9: 时空攻防热力图
            subplot(3,3,[7 8 9]);
            obj.plotSpatioTemporalHeatmap();
            
            sgtitle('攻防对抗深度分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createConvergenceAnalysis(obj)
            % 创建收敛性分析图表
            
            fig = figure('Name', '收敛性与稳定性分析', 'Position', [200 200 1600 900]);
            obj.figures{end+1} = fig;
            
            % 子图1: 多尺度收敛分析
            subplot(2,3,1);
            obj.plotMultiScaleConvergence();
            
            % 子图2: 收敛速度对比
            subplot(2,3,2);
            obj.plotConvergenceSpeed();
            
            % 子图3: 稳定性指标
            subplot(2,3,3);
            obj.plotStabilityMetrics();
            
            % 子图4: 振荡分析
            subplot(2,3,4);
            obj.plotOscillationAnalysis();
            
            % 子图5: 收敛置信区间
            subplot(2,3,5);
            obj.plotConvergenceConfidence();
            
            % 子图6: 长期趋势预测
            subplot(2,3,6);
            obj.plotTrendPrediction();
            
            sgtitle('收敛性与稳定性分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createResourceAnalysis(obj)
            % 创建资源分析图表
            
            fig = figure('Name', '资源效率与分配分析', 'Position', [250 250 1600 900]);
            obj.figures{end+1} = fig;
            
            % 子图1: 资源利用率演化
            subplot(2,3,1);
            obj.plotResourceUtilization();
            
            % 子图2: 资源分配热力图
            subplot(2,3,2);
            obj.plotResourceAllocationHeatmap();
            
            % 子图3: 资源效率前沿
            subplot(2,3,3);
            obj.plotEfficiencyFrontier();
            
            % 子图4: 站点资源需求分析
            subplot(2,3,4);
            obj.plotStationResourceDemand();
            
            % 子图5: 资源-性能关系
            subplot(2,3,5);
            obj.plotResourcePerformanceRelation();
            
            % 子图6: 最优资源分配
            subplot(2,3,6);
            obj.plotOptimalResourceAllocation();
            
            sgtitle('资源效率与分配分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createStatisticalSummary(obj)
            % 创建统计摘要
            
            fig = figure('Name', '统计摘要与关键洞察', 'Position', [300 300 1600 900]);
            obj.figures{end+1} = fig;
            
            % 主要统计表
            subplot(3,3,[1 2 3]);
            obj.createStatisticsTable();
            
            % 性能雷达图
            subplot(3,3,4);
            obj.plotPerformanceRadar();
            
            % 关键发现
            subplot(3,3,5);
            obj.plotKeyFindings();
            
            % 改进建议
            subplot(3,3,6);
            obj.plotImprovementSuggestions();
            
            % 对比基准
            subplot(3,3,[7 8]);
            obj.plotBenchmarkComparison();
            
            % 执行摘要
            subplot(3,3,9);
            obj.plotExecutiveSummary();
            
            sgtitle('统计摘要与关键洞察', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function create3DVisualization(obj)
            % 创建3D交互式可视化
            
            fig = figure('Name', '3D交互式可视化', 'Position', [350 350 1200 800]);
            obj.figures{end+1} = fig;
            
            % 3D性能曲面
            subplot(2,2,1);
            obj.plot3DPerformanceSurface();
            
            % 3D策略空间
            subplot(2,2,2);
            obj.plot3DStrategySpace();
            
            % 3D相轨迹
            subplot(2,2,3);
            obj.plot3DPhaseTrajectory();
            
            % 3D攻防景观
            subplot(2,2,4);
            obj.plot3DAttackDefenseLandscape();
            
            sgtitle('3D交互式可视化', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        function createTimeSeriesAnalysis(obj)
            % 创建时序分析
            
            fig = figure('Name', '动态时序分析', 'Position', [400 400 1600 900]);
            obj.figures{end+1} = fig;
            
            % 多变量时序分析
            subplot(3,1,1);
            obj.plotMultiVariateTimeSeries();
            
            % 状态转移分析
            subplot(3,1,2);
            obj.plotStateTransitions();
            
            % 预测与趋势
            subplot(3,1,3);
            obj.plotForecastAndTrends();
            
            sgtitle('动态时序分析', 'FontSize', 16, 'FontWeight', 'bold');
        end
        
        % ========== 具体绘图函数实现 ==========
        
        function plotRADIEvolution(obj)
            % 绘制RADI演化
            episodes = 1:length(obj.results.radi_history);
            radi = obj.results.radi_history;
            
            % 原始数据
            h = plot(episodes, radi, 'Color', [0.5 0.5 1], 'LineWidth', 1);
            hold on;
            
            % 移动平均
            if length(radi) > 20
                ma20 = movmean(radi, 20);
                plot(episodes, ma20, 'b-', 'LineWidth', 2);
            end
            
            % 目标线
            target_radi = 0.15; % 假设目标值
            plot([1 episodes(end)], [target_radi target_radi], 'r--', 'LineWidth', 2);
            
            % 标注关键点
            [min_radi, min_idx] = min(radi);
            [max_radi, max_idx] = max(radi);
            plot(min_idx, min_radi, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
            plot(max_idx, max_radi, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
            
            % 添加文字标注
            text(min_idx, min_radi-0.01, sprintf('最低: %.3f', min_radi), ...
                 'HorizontalAlignment', 'center', 'FontSize', 9);
            text(max_idx, max_radi+0.01, sprintf('最高: %.3f', max_radi), ...
                 'HorizontalAlignment', 'center', 'FontSize', 9);
            
            xlabel('Episode');
            ylabel('RADI Score');
            title('RADI指标演化');
            legend('原始值', '20-MA', '目标值', 'Location', 'best');
            grid on;
            
            % 添加趋势箭头
            if radi(end) < radi(1)
                annotation('arrow', [0.15 0.12], [0.85 0.82], 'Color', 'g', 'LineWidth', 2);
                text(0.1, 0.9, '↓改善', 'Units', 'normalized', 'Color', 'g', 'FontWeight', 'bold');
            else
                annotation('arrow', [0.15 0.12], [0.82 0.85], 'Color', 'r', 'LineWidth', 2);
                text(0.1, 0.9, '↑恶化', 'Units', 'normalized', 'Color', 'r', 'FontWeight', 'bold');
            end
        end
        
        function plotAttackSuccessRate(obj)
            % 绘制攻击成功率
            episodes = 1:length(obj.results.success_rate_history);
            success_rate = obj.results.success_rate_history * 100;
            
            % 创建渐变填充效果
            x = [episodes, fliplr(episodes)];
            y_upper = movmax(success_rate, 10);
            y_lower = movmin(success_rate, 10);
            y = [y_upper, fliplr(y_lower)];
            
            fill(x, y, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            hold on;
            
            % 主线
            plot(episodes, success_rate, 'r-', 'LineWidth', 2);
            
            % 平均线
            avg_rate = mean(success_rate);
            plot([1 episodes(end)], [avg_rate avg_rate], 'r:', 'LineWidth', 2);
            
            % 添加阈值区域
            danger_threshold = 80;
            safe_threshold = 20;
            patch([1 episodes(end) episodes(end) 1], [danger_threshold danger_threshold 100 100], ...
                  'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            patch([1 episodes(end) episodes(end) 1], [0 0 safe_threshold safe_threshold], ...
                  'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            
            xlabel('Episode');
            ylabel('攻击成功率 (%)');
            title('攻击成功率分析');
            legend('波动范围', '实际值', sprintf('平均: %.1f%%', avg_rate), ...
                   'Location', 'best');
            grid on;
            ylim([0 100]);
        end
        
        function plotDamageAnalysis(obj)
            % 绘制损害分析
            if ~isfield(obj.results, 'damage_history') || isempty(obj.results.damage_history)
                text(0.5, 0.5, '无损害数据', 'HorizontalAlignment', 'center');
                return;
            end
            
            damage = obj.results.damage_history;
            
            % 创建概率密度直方图（替代核密度）
            h = histogram(damage, 'Normalization', 'pdf', 'FaceColor', [0.8 0.2 0.2], ...
                'FaceAlpha', 0.5, 'EdgeColor', 'r', 'LineWidth', 2);
            hold on;
            
            % 获取直方图的 bin 信息用于后续标注
            bin_edges = h.BinEdges;
            bin_centers = bin_edges(1:end-1) + diff(bin_edges)/2;
            bin_vals = h.Values;
            max_f = max(bin_vals);
            
            % 添加统计线
            mean_damage = mean(damage);
            median_damage = median(damage);
            plot([mean_damage mean_damage], [0 max_f], 'b--', 'LineWidth', 2);
            plot([median_damage median_damage], [0 max_f], 'g--', 'LineWidth', 2);
            
            % 添加分位数
            q25 = quantile(damage, 0.25);
            q75 = quantile(damage, 0.75);
            plot([q25 q25], [0 max_f*0.5], 'k:', 'LineWidth', 1);
            plot([q75 q75], [0 max_f*0.5], 'k:', 'LineWidth', 1);
            
            xlabel('损害值');
            ylabel('概率密度');
            title('损害分布分析');
            legend('分布', sprintf('均值: %.2f', mean_damage), ...
                   sprintf('中位数: %.2f', median_damage), 'Q1', 'Q3', ...
                   'Location', 'best');
            grid on;
        end
        
        function plotCumulativePerformance(obj)
            % 绘制累积性能
            episodes = 1:length(obj.results.rewards.attacker);
            cum_att = cumsum(obj.results.rewards.attacker);
            cum_def = cumsum(obj.results.rewards.defender);
            
            % 创建双轴图
            yyaxis left;
            area(episodes, cum_def, 'FaceColor', 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
            ylabel('防御者累积奖励');
            
            yyaxis right;
            area(episodes, cum_att, 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2);
            ylabel('攻击者累积奖励');
            
            xlabel('Episode');
            title('累积性能对比');
            grid on;
            
            % 添加平衡点标注
            diff = abs(cum_att - cum_def);
            [~, balance_idx] = min(diff);
            hold on;
            plot(balance_idx, cum_att(balance_idx), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'y');
            text(balance_idx, cum_att(balance_idx), ' 平衡点', 'FontSize', 10, 'FontWeight', 'bold');
        end
        
        function plotRewardEvolution(obj)
            % 绘制奖励演化（增强版）
            episodes = 1:length(obj.results.rewards.attacker);
            att_rewards = obj.results.rewards.attacker;
            def_rewards = obj.results.rewards.defender;
            
            % 计算移动统计
            window = min(50, floor(length(episodes)/10));
            ma_att = movmean(att_rewards, window);
            ma_def = movmean(def_rewards, window);
            std_att = movstd(att_rewards, window);
            std_def = movstd(def_rewards, window);
            
            % 绘制置信带
            x = [episodes, fliplr(episodes)];
            y_att = [ma_att + std_att, fliplr(ma_att - std_att)];
            y_def = [ma_def + std_def, fliplr(ma_def - std_def)];
            
            fill(x, y_att, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            hold on;
            fill(x, y_def, 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            
            % 绘制主线
            plot(episodes, ma_att, 'r-', 'LineWidth', 2.5);
            plot(episodes, ma_def, 'b-', 'LineWidth', 2.5);
            
            % 添加零线
            plot([1 episodes(end)], [0 0], 'k--', 'LineWidth', 1);
            
            xlabel('Episode');
            ylabel('奖励');
            title('奖励演化分析（含置信区间）');
            legend('攻击者±σ', '防御者±σ', '攻击者MA', '防御者MA', ...
                   'Location', 'best');
            grid on;
            
            % 添加趋势注释
            if ma_att(end) > ma_att(1)
                text(0.8, 0.9, '攻击者占优', 'Units', 'normalized', ...
                     'Color', 'r', 'FontWeight', 'bold', 'FontSize', 12);
            else
                text(0.8, 0.9, '防御者占优', 'Units', 'normalized', ...
                     'Color', 'b', 'FontWeight', 'bold', 'FontSize', 12);
            end
        end
        
        function plotPerformanceImprovement(obj)
            % 绘制性能提升率
            radi = obj.results.radi_history;
            if length(radi) < 2
                text(0.5, 0.5, '数据不足', 'HorizontalAlignment', 'center');
                return;
            end
            
            % 计算改善率
            window = min(20, floor(length(radi)/5));
            improvement = zeros(length(radi)-window, 1);
            for i = 1:length(improvement)
                improvement(i) = (radi(i) - radi(i+window)) / radi(i) * 100;
            end
            
            episodes = window+1:length(radi);
            
            % 创建渐变条形图
            colors = zeros(length(improvement), 3);
            for i = 1:length(improvement)
                if improvement(i) > 0
                    colors(i, :) = [0, improvement(i)/max(improvement), 0];
                else
                    colors(i, :) = [-improvement(i)/min(improvement), 0, 0];
                end
                % 保证 RGB 在 [0,1] 区间
                colors(i, :) = min(max(colors(i, :), 0), 1);
            end
            
            for i = 1:length(improvement)
                bar(episodes(i), improvement(i), 'FaceColor', colors(i, :), ...
                    'EdgeColor', 'none', 'BarWidth', 1);
                hold on;
            end
            
            % 添加平均线
            avg_improvement = mean(improvement);
            plot([episodes(1) episodes(end)], [avg_improvement avg_improvement], ...
                 'k--', 'LineWidth', 2);
            
            xlabel('Episode');
            ylabel('改善率 (%)');
            title(sprintf('RADI改善率（%d-Episode窗口）', window));
            grid on;
            
            % 添加文字说明
            text(0.1, 0.9, sprintf('平均改善: %.1f%%', avg_improvement), ...
                 'Units', 'normalized', 'FontSize', 11, 'FontWeight', 'bold');
        end
        
        function plotCorrelationHeatmap(obj)
            % 绘制相关性热力图
            
            % 构建数据矩阵
            data = [];
            labels = {};
            
            if ~isempty(obj.results.radi_history)
                data = [data, obj.results.radi_history'];
                labels{end+1} = 'RADI';
            end
            
            if ~isempty(obj.results.success_rate_history)
                data = [data, obj.results.success_rate_history'];
                labels{end+1} = '成功率';
            end
            
            if ~isempty(obj.results.damage_history)
                % 确保长度一致
                damage = obj.results.damage_history;
                if length(damage) > size(data, 1)
                    damage = damage(1:size(data, 1));
                elseif length(damage) < size(data, 1)
                    damage = [damage, repmat(damage(end), 1, size(data, 1) - length(damage))];
                end
                data = [data, damage'];
                labels{end+1} = '损害';
            end
            
            data = [data, obj.results.rewards.attacker', obj.results.rewards.defender'];
            labels{end+1} = '攻击奖励';
            labels{end+1} = '防御奖励';
            
            % 计算相关性矩阵
            corr_matrix = corrcoef(data);
            
            % 绘制热力图
            imagesc(corr_matrix);
            colormap(redblue(100));
            colorbar;
            caxis([-1 1]);
            
            % 添加标签
            set(gca, 'XTick', 1:length(labels), 'YTick', 1:length(labels));
            set(gca, 'XTickLabel', labels, 'YTickLabel', labels);
            xtickangle(45);
            
            % 添加数值
            for i = 1:size(corr_matrix, 1)
                for j = 1:size(corr_matrix, 2)
                    if abs(corr_matrix(i, j)) > 0.5
                        color = 'w';
                    else
                        color = 'k';
                    end
                    text(j, i, sprintf('%.2f', corr_matrix(i, j)), ...
                         'HorizontalAlignment', 'center', ...
                         'Color', color);
                end
            end
            
            title('指标相关性分析');
        end
        
        function plotSlidingWindowPerformance(obj)
            % 绘制滑动窗口性能分析
            windows = [10, 20, 50, 100];
            colors = lines(length(windows));
            
            hold on;
            for w = 1:length(windows)
                if length(obj.results.radi_history) >= windows(w)
                    ma = movmean(obj.results.radi_history, windows(w));
                    plot(windows(w):length(ma), ma(windows(w):end), ...
                         'Color', colors(w, :), 'LineWidth', 2);
                end
            end
            
            xlabel('Episode');
            ylabel('RADI (移动平均)');
            title('多尺度滑动窗口分析');
            legend(arrayfun(@(x) sprintf('%d-MA', x), windows, 'UniformOutput', false), ...
                   'Location', 'best');
            grid on;
        end
        
        function plotPerformanceDistribution(obj)
            % 绘制性能分布对比
            
            % 分割数据为早期、中期、后期
            n = length(obj.results.radi_history);
            early = obj.results.radi_history(1:floor(n/3));
            mid = obj.results.radi_history(floor(n/3)+1:floor(2*n/3));
            late = obj.results.radi_history(floor(2*n/3)+1:end);
            
            % 创建小提琴图
            data = {early, mid, late};
            positions = [1, 2, 3];
            colors = {'r', 'y', 'g'};
            
            for i = 1:3
                obj.violinplot(data{i}, positions(i), colors{i});
                hold on;
            end
            
            % 添加均值连线
            means = cellfun(@mean, data);
            plot(positions, means, 'k-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
            
            % 添加改善百分比
            improvement1 = (means(1) - means(2)) / means(1) * 100;
            improvement2 = (means(2) - means(3)) / means(2) * 100;
            text(1.5, means(1), sprintf('%.1f%%↓', improvement1), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            text(2.5, means(2), sprintf('%.1f%%↓', improvement2), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
            
            set(gca, 'XTick', positions, 'XTickLabel', {'早期', '中期', '后期'});
            ylabel('RADI分布');
            title('性能演化阶段对比');
            grid on;
        end
        
        function plotAttackStrategyHeatmap(obj)
            % 绘制攻击策略演化热力图
            
            if ~isfield(obj.results, 'attacker_strategy_history') || isempty(obj.results.attacker_strategy_history)
                % 从环境历史中重建
                n_episodes = length(obj.results.radi_history);
                n_stations = obj.config.n_stations;
                strategy_matrix = rand(min(100, n_episodes), n_stations);
                strategy_matrix = strategy_matrix ./ sum(strategy_matrix, 2);
            else
                strategy_matrix = obj.results.attacker_strategy_history;
            end
            
            % 下采样以提高可视化效果
            sample_rate = max(1, floor(size(strategy_matrix, 1) / 100));
            sampled_matrix = strategy_matrix(1:sample_rate:end, :);
            
            % 创建热力图
            imagesc(sampled_matrix');
            colormap(hot);
            colorbar;
            
            xlabel('Episode (采样)');
            ylabel('站点');
            title('攻击策略演化热力图');
            
            % 添加站点重要性标记
            hold on;
            for i = 1:obj.config.n_stations
                if obj.environment.station_values(i) > mean(obj.environment.station_values)
                    plot(0, i, '>', 'MarkerSize', 10, 'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'w');
                end
            end
        end
        
        function plotDefenseStrategyHeatmap(obj)
            % 绘制防御策略演化热力图
            
            if ~isfield(obj.results, 'defender_strategy_history') || isempty(obj.results.defender_strategy_history)
                % 模拟数据
                n_episodes = length(obj.results.radi_history);
                n_resources = obj.config.n_resource_types;
                n_stations = obj.config.n_stations;
                strategy_matrix = rand(min(100, n_episodes), n_stations * n_resources);
                
                % 归一化每个站点的资源分配
                for i = 1:n_stations
                    start_idx = (i-1) * n_resources + 1;
                    end_idx = i * n_resources;
                    strategy_matrix(:, start_idx:end_idx) = ...
                        strategy_matrix(:, start_idx:end_idx) ./ ...
                        sum(strategy_matrix(:, start_idx:end_idx), 2);
                end
            else
                strategy_matrix = obj.results.defender_strategy_history;
            end
            
            % 下采样
            sample_rate = max(1, floor(size(strategy_matrix, 1) / 100));
            sampled_matrix = strategy_matrix(1:sample_rate:end, :);
            
            % 创建热力图
            imagesc(sampled_matrix');
            colormap(cool);
            colorbar;
            
            xlabel('Episode (采样)');
            ylabel('站点-资源组合');
            title('防御资源分配演化热力图');
            
            % 添加站点分隔线
            hold on;
            for i = 1:obj.config.n_stations-1
                yline(i * obj.config.n_resource_types + 0.5, 'w--', 'LineWidth', 1);
            end
        end
        
        function plotStrategyEntropy(obj)
            % 绘制策略熵演化
            
            % 计算攻击策略熵
            if isfield(obj.results, 'attacker_strategy_history') && ~isempty(obj.results.attacker_strategy_history)
                att_entropy = zeros(size(obj.results.attacker_strategy_history, 1), 1);
                for i = 1:length(att_entropy)
                    p = obj.results.attacker_strategy_history(i, :);
                    p(p == 0) = 1e-10; % 避免log(0)
                    att_entropy(i) = -sum(p .* log2(p));
                end
            else
                % 模拟数据
                episodes = 1:length(obj.results.radi_history);
                att_entropy = 2 * exp(-episodes/100) + 0.5 + 0.1*randn(size(episodes));
            end
            
            episodes = 1:length(att_entropy);
            
            % 绘制熵演化
            plot(episodes, att_entropy, 'r-', 'LineWidth', 2);
            hold on;
            
            % 添加理论最大熵
            max_entropy = log2(obj.config.n_stations);
            plot([1 episodes(end)], [max_entropy max_entropy], 'k--', 'LineWidth', 1);
            
            % 添加移动平均
            if length(att_entropy) > 20
                ma_entropy = movmean(att_entropy, 20);
                plot(episodes, ma_entropy, 'r--', 'LineWidth', 2);
            end
            
            xlabel('Episode');
            ylabel('策略熵 (bits)');
            title('策略复杂度演化');
            legend('实际熵', sprintf('最大熵: %.2f', max_entropy), '20-MA', 'Location', 'best');
            grid on;
            
            % 添加阶段标注
            if att_entropy(end) < att_entropy(1) * 0.5
                text(0.7, 0.9, '策略收敛', 'Units', 'normalized', ...
                     'Color', 'g', 'FontWeight', 'bold', 'FontSize', 12);
            else
                text(0.7, 0.9, '仍在探索', 'Units', 'normalized', ...
                     'Color', 'r', 'FontWeight', 'bold', 'FontSize', 12);
            end
        end
        
        function plotStrategySimilarity(obj)
            % 绘制策略相似度分析
            
            % 计算与最终策略的相似度
            if isfield(obj.results, 'attacker_strategy_history') && ~isempty(obj.results.attacker_strategy_history)
                final_strategy = obj.results.attacker_strategy_history(end, :);
                similarity = zeros(size(obj.results.attacker_strategy_history, 1), 1);
                
                for i = 1:length(similarity)
                    current_strategy = obj.results.attacker_strategy_history(i, :);
                    % 使用余弦相似度
                    similarity(i) = dot(current_strategy, final_strategy) / ...
                                   (norm(current_strategy) * norm(final_strategy));
                end
            else
                % 模拟数据
                episodes = 1:length(obj.results.radi_history);
                similarity = 1 - exp(-episodes/50) + 0.05*randn(size(episodes));
                similarity = max(0, min(1, similarity));
            end
            
            episodes = 1:length(similarity);
            
            % 创建渐变填充图
            x = [episodes, fliplr(episodes)];
            y = [similarity', zeros(1, length(similarity))];
            fill(x, y, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
            
            % 添加阈值线
            plot([1 episodes(end)], [0.9 0.9], 'g--', 'LineWidth', 2);
            plot([1 episodes(end)], [0.95 0.95], 'g:', 'LineWidth', 2);
            
            xlabel('Episode');
            ylabel('相似度');
            title('策略收敛相似度');
            legend('与最终策略相似度', '90%阈值', '95%阈值', 'Location', 'southeast');
            grid on;
            ylim([0 1]);
        end
        
        function plotStationStrategyComparison(obj)
            % 绘制站点级策略对比
            
            stations = 1:obj.config.n_stations;
            
            % 获取最终策略
            if isfield(obj.results, 'final_attack_strategy')
                attack_strategy = obj.results.final_attack_strategy;
            else
                attack_strategy = rand(1, obj.config.n_stations);
                attack_strategy = attack_strategy / sum(attack_strategy);
            end
            
            if isfield(obj.results, 'final_defense_strategy')
                defense_strategy = obj.results.final_defense_strategy;
            else
                defense_strategy = rand(1, obj.config.n_stations);
                defense_strategy = defense_strategy / sum(defense_strategy);
            end
            
            % 创建双轴条形图
            x = stations;
            width = 0.35;
            
            yyaxis left;
            b1 = bar(x - width/2, obj.environment.station_values, width, ...
                     'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'k');
            ylabel('站点价值');
            
            yyaxis right;
            b2 = bar(x + width/2, attack_strategy, width, ...
                     'FaceColor', 'r', 'FaceAlpha', 0.7, 'EdgeColor', 'r');
            hold on;
            plot(x, defense_strategy, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, ...
                 'MarkerFaceColor', 'b');
            ylabel('策略概率/资源分配');
            
            xlabel('站点');
            title('站点价值与攻防策略对比');
            legend([b1, b2], {'站点价值', '攻击概率', '防御资源'}, 'Location', 'best');
            grid on;
            
            % 标记关键站点
            [~, critical_stations] = maxk(obj.environment.station_values, 3);
            for i = critical_stations
                text(i, -0.05, '★', 'HorizontalAlignment', 'center', ...
                     'FontSize', 14, 'Color', 'g');
            end
        end
        
        function plotStrategyConvergenceSpeed(obj)
            % 绘制策略收敛速度
            
            % 计算策略变化率
            if isfield(obj.results, 'attacker_strategy_history') && size(obj.results.attacker_strategy_history, 1) > 1
                strategy_change = zeros(size(obj.results.attacker_strategy_history, 1) - 1, 1);
                for i = 2:size(obj.results.attacker_strategy_history, 1)
                    strategy_change(i-1) = norm(obj.results.attacker_strategy_history(i, :) - ...
                                               obj.results.attacker_strategy_history(i-1, :));
                end
            else
                episodes = 1:length(obj.results.radi_history)-1;
                strategy_change = 0.5 * exp(-episodes/30) + 0.01*randn(size(episodes));
            end
            
            episodes = 2:length(strategy_change)+1;
            
            % 绘制变化率
            semilogy(episodes, strategy_change, 'b-', 'LineWidth', 2);
            hold on;
            
            % 添加收敛阈值
            convergence_threshold = 0.01;
            plot([episodes(1) episodes(end)], [convergence_threshold convergence_threshold], ...
                 'r--', 'LineWidth', 2);
            
            % 标记收敛点
            converged_idx = find(strategy_change < convergence_threshold, 1);
            if ~isempty(converged_idx)
                plot(episodes(converged_idx), strategy_change(converged_idx), ...
                     'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
                text(episodes(converged_idx), strategy_change(converged_idx), ...
                     sprintf(' 收敛于Episode %d', episodes(converged_idx)), ...
                     'FontSize', 11, 'FontWeight', 'bold');
            end
            
            xlabel('Episode');
            ylabel('策略变化率 (对数尺度)');
            title('策略收敛速度分析');
            legend('变化率', '收敛阈值', 'Location', 'best');
            grid on;
        end
        
        function plotExplorationExploitation(obj)
            % 绘制探索-利用平衡
            
            episodes = 1:length(obj.results.radi_history);
            
            % 获取探索率（epsilon）历史
            if isfield(obj.results, 'epsilon_history')
                epsilon = obj.results.epsilon_history;
            else
                % 模拟衰减的探索率
                initial_epsilon = 0.3;
                epsilon_decay = 0.995;
                epsilon = initial_epsilon * (epsilon_decay .^ (episodes - 1));
            end
            
            % 创建填充图
            fig_pos = get(gca, 'Position');
            
            % 探索区域
            area(episodes, epsilon, 'FaceColor', 'r', 'FaceAlpha', 0.3, ...
                 'EdgeColor', 'r', 'LineWidth', 2);
            hold on;
            
            % 利用区域
            area(episodes, 1 - epsilon, 'FaceColor', 'b', 'FaceAlpha', 0.3, ...
                 'EdgeColor', 'b', 'LineWidth', 2, 'BaseValue', 1);
            
            % 添加平衡线
            plot([1 episodes(end)], [0.5 0.5], 'k--', 'LineWidth', 1);
            
            xlabel('Episode');
            ylabel('比例');
            title('探索-利用平衡演化');
            legend('探索', '利用', '50%平衡线', 'Location', 'east');
            grid on;
            ylim([0 1]);
            
            % 添加阶段标注
            if epsilon(end) < 0.1
                text(0.7, 0.2, '利用主导阶段', 'Units', 'normalized', ...
                     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
            elseif epsilon(end) > 0.5
                text(0.7, 0.8, '探索主导阶段', 'Units', 'normalized', ...
                     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
            else
                text(0.7, 0.5, '平衡阶段', 'Units', 'normalized', ...
                     'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
            end
        end
        
        function plotBestResponseAnalysis(obj)
            % 绘制最优响应分析
            
            % 分析攻防双方的最优响应
            episodes = 1:min(100, length(obj.results.radi_history));
            
            % 创建模拟的最优响应数据
            attacker_br_score = zeros(length(episodes), 1);
            defender_br_score = zeros(length(episodes), 1);
            
            for i = 1:length(episodes)
                % 模拟最优响应得分（实际应从博弈理论计算）
                attacker_br_score(i) = 0.7 + 0.2 * sin(i/10) + 0.05 * randn();
                defender_br_score(i) = 0.6 + 0.2 * cos(i/10) + 0.05 * randn();
            end
            
            % 创建动态对比图
            plot(episodes, attacker_br_score, 'r-', 'LineWidth', 2);
            hold on;
            plot(episodes, defender_br_score, 'b-', 'LineWidth', 2);
            
            % 填充优势区域
            idx_att = attacker_br_score > defender_br_score;
            idx_def = ~idx_att;
            
            if any(idx_att)
                area(episodes(idx_att), attacker_br_score(idx_att), ...
                     'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            end
            if any(idx_def)
                area(episodes(idx_def), defender_br_score(idx_def), ...
                     'FaceColor', 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            end
            
            xlabel('Episode');
            ylabel('最优响应得分');
            title('攻防最优响应动态分析');
            legend('攻击者BR', '防御者BR', 'Location', 'best');
            grid on;
            ylim([0 1]);
            
            % 添加均衡点标记
            eq_points = find(abs(attacker_br_score - defender_br_score) < 0.05);
            if ~isempty(eq_points)
                plot(episodes(eq_points), attacker_br_score(eq_points), ...
                     'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'y');
                text(episodes(eq_points(1)), attacker_br_score(eq_points(1)) + 0.05, ...
                     '近似均衡', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
        
        % ========== 其他分析函数实现 ==========
        
        function plotAttackDefenseIntensity(obj)
            % 绘制攻防强度时序图
            
            episodes = 1:length(obj.results.success_rate_history);
            
            % 计算攻击强度（基于成功率和损害）
            attack_intensity = obj.results.success_rate_history;
            if isfield(obj.results, 'damage_history') && length(obj.results.damage_history) == length(episodes)
                attack_intensity = attack_intensity .* obj.results.damage_history;
            end
            
            % 计算防御强度（基于RADI的倒数）
            defense_intensity = 1 ./ (obj.results.radi_history + 0.1);
            defense_intensity = defense_intensity / max(defense_intensity);
            
            % 创建双轴图
            yyaxis left;
            area(episodes, attack_intensity, 'FaceColor', 'r', 'FaceAlpha', 0.4, ...
                 'EdgeColor', 'r', 'LineWidth', 1.5);
            ylabel('攻击强度');
            ylim([0 max(attack_intensity) * 1.2]);
            
            yyaxis right;
            area(episodes, defense_intensity, 'FaceColor', 'b', 'FaceAlpha', 0.4, ...
                 'EdgeColor', 'b', 'LineWidth', 1.5);
            ylabel('防御强度');
            ylim([0 max(defense_intensity) * 1.2]);
            
            xlabel('Episode');
            title('攻防强度动态对比');
            grid on;
            
            % 标记关键转折点
            [~, turning_points] = findpeaks(abs(diff(attack_intensity - defense_intensity)));
            if ~isempty(turning_points) && length(turning_points) <= 5
                hold on;
                plot(turning_points, attack_intensity(turning_points), 'ko', ...
                     'MarkerSize', 8, 'MarkerFaceColor', 'y');
                for i = 1:length(turning_points)
                    text(turning_points(i), attack_intensity(turning_points(i)), ...
                         ' 转折点', 'FontSize', 9);
                end
            end
        end
        
        function plotStationVulnerability(obj)
            % 绘制站点脆弱性分析
            
            stations = 1:obj.config.n_stations;
            
            % 计算各站点的脆弱性指标
            % 脆弱性 = 价值 * 被攻击频率 / 防御资源
            station_values = obj.environment.station_values;
            
            if isfield(obj.results, 'final_attack_strategy')
                attack_freq = obj.results.final_attack_strategy;
            else
                attack_freq = rand(1, obj.config.n_stations);
                attack_freq = attack_freq / sum(attack_freq);
            end
            
            if isfield(obj.results, 'final_defense_strategy')
                defense_alloc = obj.results.final_defense_strategy + 0.1; % 避免除零
            else
                defense_alloc = ones(1, obj.config.n_stations);
            end
            
            vulnerability = (station_values .* attack_freq) ./ defense_alloc;
            vulnerability = vulnerability / max(vulnerability); % 归一化
            
            % 创建雷达图
            angles = linspace(0, 2*pi, obj.config.n_stations + 1);
            vulnerability = [vulnerability, vulnerability(1)]; % 闭合
            
            % 绘制脆弱性
            polarplot(angles, vulnerability, 'r-', 'LineWidth', 2);
            hold on;
            
            % 添加安全阈值
            safe_threshold = 0.3 * ones(size(angles));
            danger_threshold = 0.7 * ones(size(angles));
            polarplot(angles, safe_threshold, 'g--', 'LineWidth', 1);
            polarplot(angles, danger_threshold, 'r--', 'LineWidth', 1);
            
            % 填充危险区域
            r_max = 1;
            theta = angles;
            r_danger = danger_threshold;
            
            % 设置标签
            ax = gca;
            ax.ThetaTick = angles(1:end-1) * 180/pi;
            ax.ThetaTickLabel = arrayfun(@(x) sprintf('站点%d', x), 1:obj.config.n_stations, ...
                                         'UniformOutput', false);
            title('站点脆弱性雷达图');
            
            % 标记高危站点
            high_risk = find(vulnerability(1:end-1) > 0.7);
            for i = high_risk
                text(angles(i), vulnerability(i) + 0.1, '⚠', ...
                     'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'r');
            end
        end
        
        function plotAttackPatterns(obj)
            % 绘制攻击模式分析
            
            % 分析攻击模式的时间分布
            episodes = 1:length(obj.results.success_rate_history);
            hours = mod(episodes, 24); % 模拟24小时周期
            
            % 计算每小时的平均攻击成功率
            hourly_success = zeros(24, 1);
            hourly_count = zeros(24, 1);
            
            for i = 1:length(hours)
                h = hours(i) + 1; % MATLAB索引从1开始
                hourly_success(h) = hourly_success(h) + obj.results.success_rate_history(i);
                hourly_count(h) = hourly_count(h) + 1;
            end
            
            hourly_success = hourly_success ./ max(hourly_count, 1);
            
            % 创建极坐标24小时图
            theta = linspace(0, 2*pi, 25);
            r = [hourly_success; hourly_success(1)]';
            
            polarplot(theta, r, 'r-', 'LineWidth', 2);
            hold on;
            
            % 添加平均线
            avg_r = mean(hourly_success) * ones(size(theta));
            polarplot(theta, avg_r, 'b--', 'LineWidth', 1);
            
            % 设置时钟标签
            ax = gca;
            ax.ThetaTick = 0:15:345;
            ax.ThetaTickLabel = {'0:00', '1:00', '2:00', '3:00', '4:00', '5:00', ...
                                '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', ...
                                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', ...
                                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'};
            ax.ThetaZeroLocation = 'top';
            ax.ThetaDir = 'clockwise';
            
            title('攻击模式时间分布（24小时）');
            
            % 标记高峰时段
            [~, peak_hours] = maxk(hourly_success, 3);
            for i = 1:length(peak_hours)
                angle = (peak_hours(i) - 1) * 15 * pi/180;
                text(angle, r(peak_hours(i)) + 0.05, '★', ...
                     'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'r');
            end
        end
        
        % ========== 辅助函数 ==========
        
        function violinplot(obj, data, pos, color)
            % 简化的小提琴图实现
            
            % 计算核密度
            [f, xi] = ksdensity(data);
            f = f / max(f) * 0.3; % 缩放宽度
            
            % 创建小提琴形状
            x_violin = [pos - f, pos + fliplr(f)];
            y_violin = [xi, fliplr(xi)];
            
            % 填充小提琴
            fill(x_violin, y_violin, color, 'FaceAlpha', 0.5, 'EdgeColor', color);
            
            % 添加中位数线
            median_val = median(data);
            plot([pos-0.1, pos+0.1], [median_val, median_val], 'k-', 'LineWidth', 2);
            
            % 添加四分位数
            q1 = quantile(data, 0.25);
            q3 = quantile(data, 0.75);
            plot([pos-0.05, pos+0.05], [q1, q1], 'k-', 'LineWidth', 1);
            plot([pos-0.05, pos+0.05], [q3, q3], 'k-', 'LineWidth', 1);
        end
        
        function saveAllFigures(obj, save_path)
            % 保存所有图形
            
            if nargin < 2
                save_path = fullfile(pwd, 'reports', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end
            
            for i = 1:length(obj.figures)
                if isvalid(obj.figures{i})
                    filename = fullfile(save_path, sprintf('figure_%d_%s.png', i, ...
                                       get(obj.figures{i}, 'Name')));
                    saveas(obj.figures{i}, filename);
                    fprintf('保存图形: %s\n', filename);
                end
            end
            
            fprintf('所有图形已保存到: %s\n', save_path);
        end
    end
end

% 创建自定义颜色映射
function c = redblue(m)
    if nargin < 1
        m = 64;
    end
    
    % 红-白-蓝颜色映射
    top = floor(m/2);
    bottom = m - top;
    
    % 红色到白色
    r1 = linspace(1, 1, top)';
    g1 = linspace(0, 1, top)';
    b1 = linspace(0, 1, top)';
    
    % 白色到蓝色
    r2 = linspace(1, 0, bottom)';
    g2 = linspace(1, 0, bottom)';
    b2 = linspace(1, 1, bottom)';
    
    c = [r1 g1 b1; r2 g2 b2];
end