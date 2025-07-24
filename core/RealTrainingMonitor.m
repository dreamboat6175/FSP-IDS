%% RealTrainingMonitor.m - 真实训练过程监控器
% =========================================================================
% 功能：实时监控和记录真实的训练过程，特别关注RADI和检测率的跳变
% 使用：在主训练循环中调用 monitor.logIteration()
% =========================================================================

classdef RealTrainingMonitor < handle
    
    properties
        config
        log_file
        real_data
        stability_flags
        jump_thresholds
        convergence_window
    end
    
    methods
        function obj = RealTrainingMonitor(config)
            obj.config = config;
            obj.initializeLogging();
            obj.initializeDataStructures();
            obj.setThresholds();
        end
        
        function initializeLogging(obj)
            % 初始化真实数据日志文件
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            obj.log_file = fullfile('logs', sprintf('real_training_%s.log', timestamp));
            
            if ~exist('logs', 'dir')
                mkdir('logs');
            end
            
            % 写入日志头
            fid = fopen(obj.log_file, 'w');
            fprintf(fid, 'FSP-TCS 真实训练数据日志\n');
            fprintf(fid, '=====================\n');
            fprintf(fid, '开始时间: %s\n', datestr(now));
            fprintf(fid, '配置: %s\n', jsonencode(obj.config));
            fprintf(fid, '=====================\n\n');
            fclose(fid);
        end
        
        function initializeDataStructures(obj)
            % 初始化数据结构
            obj.real_data = struct();
            obj.real_data.iteration = [];
            obj.real_data.radi_values = [];
            obj.real_data.detection_rates = [];
            obj.real_data.rewards = [];
            obj.real_data.learning_rates = [];
            obj.real_data.epsilon_values = [];
            obj.real_data.q_value_changes = [];
            
            obj.stability_flags = struct();
            obj.stability_flags.radi_jumps = [];
            obj.stability_flags.detection_jumps = [];
            obj.stability_flags.divergence_warnings = [];
            
            obj.convergence_window = 50; % 用于收敛检测的窗口大小
        end
        
        function setThresholds(obj)
            % 设置跳变检测阈值
            obj.jump_thresholds = struct();
            obj.jump_thresholds.radi = 0.01;        % RADI跳变阈值
            obj.jump_thresholds.detection = 0.05;   % 检测率跳变阈值
            obj.jump_thresholds.reward = 0.1;       % 奖励跳变阈值
            obj.jump_thresholds.q_value = 0.5;      % Q值跳变阈值
        end
        
        function logIteration(obj, iteration, agents, performance_data)
            % 记录每次迭代的真实数据
            try
                % 记录基本信息
                obj.real_data.iteration(end+1) = iteration;
                
                % 从智能体提取真实数据
                [radi, detection, rewards, lr, epsilon, q_changes] = obj.extractIterationData(agents, performance_data);
                
                % 存储数据
                obj.real_data.radi_values(end+1) = radi;
                obj.real_data.detection_rates(end+1) = detection;
                obj.real_data.rewards(end+1) = rewards;
                obj.real_data.learning_rates(end+1) = lr;
                obj.real_data.epsilon_values(end+1) = epsilon;
                obj.real_data.q_value_changes(end+1) = q_changes;
                
                % 检测跳变和异常
                obj.detectJumps(iteration);
                
                % 评估收敛性
                obj.assessConvergence(iteration);
                
                % 实时输出关键信息
                obj.printRealTimeStatus(iteration, radi, detection);
                
                % 写入详细日志
                obj.writeDetailedLog(iteration, radi, detection, rewards, lr, epsilon);
                
            catch ME
                fprintf('❌ 监控器记录失败: %s\n', ME.message);
                obj.logError(iteration, ME);
            end
        end
        
        function [radi, detection, rewards, lr, epsilon, q_changes] = extractIterationData(obj, agents, performance_data)
            % 提取当前迭代的真实数据
            
            % 默认值
            radi = NaN;
            detection = NaN;
            rewards = NaN;
            lr = NaN;
            epsilon = NaN;
            q_changes = NaN;
            
            try
                % 从性能数据中提取RADI
                if isfield(performance_data, 'radi')
                    radi = performance_data.radi;
                elseif isfield(performance_data, 'avg_radi')
                    radi = performance_data.avg_radi;
                end
                
                % 从性能数据中提取检测率
                if isfield(performance_data, 'detection_rate')
                    detection = performance_data.detection_rate;
                elseif isfield(performance_data, 'avg_detection_rate')
                    detection = mean(performance_data.avg_detection_rate);
                end
                
                % 从智能体中提取学习参数
                if length(agents) >= 2  % 假设第二个是防御者
                    defender = agents{2};
                    
                    % 学习率
                    if isprop(defender, 'lr_scheduler') && isfield(defender.lr_scheduler, 'current_lr')
                        lr = defender.lr_scheduler.current_lr;
                    elseif isprop(defender, 'learning_rate')
                        lr = defender.learning_rate;
                    end
                    
                    % Epsilon值
                    if isprop(defender, 'epsilon')
                        epsilon = defender.epsilon;
                    end
                    
                    % Q值变化
                    if isprop(defender, 'Q_table') && ~isempty(obj.real_data.q_value_changes)
                        current_q_mean = mean(defender.Q_table(:));
                        if length(obj.real_data.q_value_changes) > 0
                            last_q_mean = obj.real_data.q_value_changes(end);
                            q_changes = abs(current_q_mean - last_q_mean);
                        else
                            q_changes = 0;
                        end
                    end
                end
                
                % 奖励数据
                if isfield(performance_data, 'avg_defender_reward')
                    rewards = mean(performance_data.avg_defender_reward);
                elseif isfield(performance_data, 'defender_reward')
                    rewards = performance_data.defender_reward;
                end
                
            catch ME
                fprintf('⚠️ 数据提取警告: %s\n', ME.message);
            end
        end
        
        function detectJumps(obj, iteration)
            % 检测数据跳变
            if iteration < 2
                return;
            end
            
            % 检测RADI跳变
            if length(obj.real_data.radi_values) >= 2
                radi_change = abs(obj.real_data.radi_values(end) - obj.real_data.radi_values(end-1));
                if radi_change > obj.jump_thresholds.radi
                    warning_msg = sprintf('🚨 RADI跳变检测！迭代%d: 变化%.4f (阈值%.4f)', ...
                        iteration, radi_change, obj.jump_thresholds.radi);
                    fprintf('%s\n', warning_msg);
                    obj.stability_flags.radi_jumps(end+1) = iteration;
                    obj.logWarning(warning_msg);
                end
            end
            
            % 检测检测率跳变
            if length(obj.real_data.detection_rates) >= 2
                det_change = abs(obj.real_data.detection_rates(end) - obj.real_data.detection_rates(end-1));
                if det_change > obj.jump_thresholds.detection
                    warning_msg = sprintf('🚨 检测率跳变检测！迭代%d: 变化%.4f (阈值%.4f)', ...
                        iteration, det_change, obj.jump_thresholds.detection);
                    fprintf('%s\n', warning_msg);
                    obj.stability_flags.detection_jumps(end+1) = iteration;
                    obj.logWarning(warning_msg);
                end
            end
            
            % 检测Q值剧烈变化
            if length(obj.real_data.q_value_changes) >= 2
                q_change = obj.real_data.q_value_changes(end);
                if q_change > obj.jump_thresholds.q_value
                    warning_msg = sprintf('🚨 Q值剧烈变化！迭代%d: 变化%.4f', iteration, q_change);
                    fprintf('%s\n', warning_msg);
                    obj.logWarning(warning_msg);
                end
            end
        end
        
        function assessConvergence(obj, iteration)
            % 评估收敛性
            if iteration < obj.convergence_window
                return;
            end
            
            % 检查RADI收敛
            if length(obj.real_data.radi_values) >= obj.convergence_window
                recent_radi = obj.real_data.radi_values(end-obj.convergence_window+1:end);
                radi_var = var(recent_radi);
                radi_trend = obj.calculateTrend(recent_radi);
                
                if radi_var < 0.0001 && abs(radi_trend) < 0.0001
                    fprintf('✅ RADI收敛检测！迭代%d (方差: %.6f, 趋势: %.6f)\n', iteration, radi_var, radi_trend);
                elseif radi_var > 0.01
                    fprintf('⚠️ RADI发散警告！迭代%d (方差: %.6f)\n', iteration, radi_var);
                    obj.stability_flags.divergence_warnings(end+1) = iteration;
                end
            end
        end
        
        function trend = calculateTrend(obj, data)
            % 计算数据趋势
            x = 1:length(data);
            p = polyfit(x, data, 1);
            trend = p(1);  % 斜率
        end
        
        function printRealTimeStatus(obj, iteration, radi, detection)
            % 实时输出状态
            if mod(iteration, 10) == 0  % 每10次迭代输出一次
                fprintf('\n--- 真实训练状态 (迭代 %d) ---\n', iteration);
                if ~isnan(radi)
                    fprintf('RADI: %.4f', radi);
                    if length(obj.real_data.radi_values) >= 2
                        change = radi - obj.real_data.radi_values(end-1);
                        fprintf(' (变化: %+.4f)', change);
                    end
                    fprintf('\n');
                end
                
                if ~isnan(detection)
                    fprintf('检测率: %.4f', detection);
                    if length(obj.real_data.detection_rates) >= 2
                        change = detection - obj.real_data.detection_rates(end-1);
                        fprintf(' (变化: %+.4f)', change);
                    end
                    fprintf('\n');
                end
                
                % 显示跳变统计
                fprintf('RADI跳变次数: %d, 检测率跳变次数: %d\n', ...
                    length(obj.stability_flags.radi_jumps), length(obj.stability_flags.detection_jumps));
                fprintf('--------------------------------\n');
            end
        end
        
        function writeDetailedLog(obj, iteration, radi, detection, rewards, lr, epsilon)
            % 写入详细日志
            fid = fopen(obj.log_file, 'a');
            fprintf(fid, '[%s] 迭代%d: RADI=%.4f, 检测率=%.4f, 奖励=%.4f, 学习率=%.6f, Epsilon=%.4f\n', ...
                datestr(now, 'yyyy-mm-dd HH:MM:SS'), iteration, radi, detection, rewards, lr, epsilon);
            fclose(fid);
        end
        
        function logWarning(obj, message)
            % 记录警告
            fid = fopen(obj.log_file, 'a');
            fprintf(fid, '[%s] 警告: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
            fclose(fid);
        end
        
        function logError(obj, iteration, ME)
            % 记录错误
            fid = fopen(obj.log_file, 'a');
            fprintf(fid, '[%s] 错误 (迭代%d): %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), iteration, ME.message);
            fclose(fid);
        end
        
        function generateRealReport(obj)
            % 生成基于真实数据的报告
            fprintf('\n📊 生成真实训练报告...\n');
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            report_file = fullfile('reports', sprintf('real_training_report_%s.txt', timestamp));
            
            if ~exist('reports', 'dir')
                mkdir('reports');
            end
            
            fid = fopen(report_file, 'w');
            
            fprintf(fid, 'FSP-TCS 真实训练分析报告\n');
            fprintf(fid, '========================\n');
            fprintf(fid, '生成时间: %s\n', datestr(now));
            fprintf(fid, '总迭代数: %d\n\n', length(obj.real_data.iteration));
            
            % RADI分析
            fprintf(fid, '=== RADI分析 ===\n');
            if ~all(isnan(obj.real_data.radi_values))
                valid_radi = obj.real_data.radi_values(~isnan(obj.real_data.radi_values));
                fprintf(fid, '平均值: %.4f\n', mean(valid_radi));
                fprintf(fid, '标准差: %.4f\n', std(valid_radi));
                fprintf(fid, '最大值: %.4f\n', max(valid_radi));
                fprintf(fid, '最小值: %.4f\n', min(valid_radi));
                fprintf(fid, '跳变次数: %d\n', length(obj.stability_flags.radi_jumps));
                if ~isempty(obj.stability_flags.radi_jumps)
                    fprintf(fid, '跳变发生在迭代: %s\n', mat2str(obj.stability_flags.radi_jumps));
                end
            else
                fprintf(fid, '❌ 没有有效的RADI数据\n');
            end
            
            % 检测率分析
            fprintf(fid, '\n=== 检测率分析 ===\n');
            if ~all(isnan(obj.real_data.detection_rates))
                valid_detection = obj.real_data.detection_rates(~isnan(obj.real_data.detection_rates));
                fprintf(fid, '平均值: %.4f\n', mean(valid_detection));
                fprintf(fid, '标准差: %.4f\n', std(valid_detection));
                fprintf(fid, '最大值: %.4f\n', max(valid_detection));
                fprintf(fid, '最小值: %.4f\n', min(valid_detection));
                fprintf(fid, '跳变次数: %d\n', length(obj.stability_flags.detection_jumps));
                if ~isempty(obj.stability_flags.detection_jumps)
                    fprintf(fid, '跳变发生在迭代: %s\n', mat2str(obj.stability_flags.detection_jumps));
                end
            else
                fprintf(fid, '❌ 没有有效的检测率数据\n');
            end
            
            % 稳定性评估
            fprintf(fid, '\n=== 稳定性评估 ===\n');
            if isempty(obj.stability_flags.radi_jumps) && isempty(obj.stability_flags.detection_jumps)
                fprintf(fid, '✅ 训练过程稳定，无显著跳变\n');
            else
                fprintf(fid, '❌ 训练过程不稳定\n');
                fprintf(fid, '建议：\n');
                fprintf(fid, '1. 降低学习率\n');
                fprintf(fid, '2. 减缓epsilon衰减\n');
                fprintf(fid, '3. 增加平滑机制\n');
                fprintf(fid, '4. 检查数据数值稳定性\n');
            end
            
            fclose(fid);
            fprintf('📝 真实训练报告已保存: %s\n', report_file);
        end
    end
end