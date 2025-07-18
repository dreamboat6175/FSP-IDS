function handleIterationOutput(iteration, config, iteration_time, episode_results)
    %% handleIterationOutput - 处理迭代输出和进度显示
    % 输入:
    %   iteration - 当前迭代编号
    %   config - 配置结构体
    %   iteration_time - 当前迭代耗时
    %   episode_results - 当前迭代的episode结果
    
    try
        % 获取显示和保存间隔
        if isfield(config, 'performance')
            display_interval = getFieldOrDefault(config.performance, 'display_interval', 50);
            save_interval = getFieldOrDefault(config.performance, 'save_interval', 100);
        else
            display_interval = 50;
            save_interval = 100;
        end
        
        % 计算关键指标
        metrics = calculateDisplayMetrics(episode_results);
        
        % 显示进度信息
        if mod(iteration, display_interval) == 0
            displayProgress(iteration, config, iteration_time, metrics);
            
            % 记录到日志
            if exist('Logger', 'class')
                Logger.info(sprintf('迭代 %d 完成，用时 %.2f秒', iteration, iteration_time));
            end
        end
        
        % 保存中间结果
        if mod(iteration, save_interval) == 0
            saveIntermediateResults(iteration, config, episode_results);
        end
        
        % 特殊里程碑显示
        displayMilestones(iteration, config);
        
        % 性能警告检查
        checkPerformanceWarnings(iteration, metrics, iteration_time);
        
        % 估算剩余时间
        if mod(iteration, display_interval) == 0
            estimateRemainingTime(iteration, config, iteration_time);
        end
        
    catch ME
        warning('处理迭代输出时出错 (迭代 %d): %s', iteration, ME.message);
        % 至少显示基本进度
        try
            if mod(iteration, 50) == 0
                fprintf('迭代 %d/%d 完成 (%.1f%%), 耗时: %.2fs\n', ...
                        iteration, config.n_iterations, ...
                        (iteration/config.n_iterations)*100, iteration_time);
            end
        catch
            % 即使基本显示也失败，静默继续
        end
    end
end

function metrics = calculateDisplayMetrics(episode_results)
    %% calculateDisplayMetrics - 计算用于显示的关键指标
    
    metrics = struct();
    
    % RADI指标
    if isfield(episode_results, 'avg_radi')
        metrics.avg_radi = mean(episode_results.avg_radi);
        metrics.max_radi = max(episode_results.avg_radi);
        metrics.min_radi = min(episode_results.avg_radi);
    else
        metrics.avg_radi = NaN;
        metrics.max_radi = NaN;
        metrics.min_radi = NaN;
    end
    
    % 成功/检测率
    if isfield(episode_results, 'attack_info')
        attack_success_rate = mean([episode_results.attack_info{:}]);
        metrics.success_rate = 1 - attack_success_rate;
        metrics.detection_rate = 1 - attack_success_rate;
    else
        metrics.success_rate = NaN;
        metrics.detection_rate = NaN;
    end
    
    % 资源效率
    if isfield(episode_results, 'avg_efficiency')
        metrics.efficiency = mean(episode_results.avg_efficiency);
    else
        metrics.efficiency = NaN;
    end
    
    % 分配均衡性
    if isfield(episode_results, 'avg_balance')
        metrics.balance = mean(episode_results.avg_balance);
    else
        metrics.balance = NaN;
    end
    
    % 奖励指标
    if isfield(episode_results, 'avg_defender_reward')
        metrics.defender_reward = mean(episode_results.avg_defender_reward);
    else
        metrics.defender_reward = NaN;
    end
    
    if isfield(episode_results, 'avg_attacker_reward')
        metrics.attacker_reward = episode_results.avg_attacker_reward;
    else
        metrics.attacker_reward = NaN;
    end
end

function displayProgress(iteration, config, iteration_time, metrics)
    %% displayProgress - 显示详细的进度信息
    
    progress_pct = (iteration / config.n_iterations) * 100;
    
    fprintf('\n=== 迭代 %d/%d (%.1f%%) ===\n', ...
            iteration, config.n_iterations, progress_pct);
    
    % 显示关键指标
    if ~isnan(metrics.avg_radi)
        fprintf('RADI: %.3f (范围: %.3f-%.3f)\n', ...
                metrics.avg_radi, metrics.min_radi, metrics.max_radi);
    end
    
    if ~isnan(metrics.detection_rate)
        fprintf('检测率: %.1f%%\n', metrics.detection_rate * 100);
    end
    
    if ~isnan(metrics.efficiency)
        fprintf('资源效率: %.1f%%\n', metrics.efficiency * 100);
    end
    
    if ~isnan(metrics.balance)
        fprintf('分配均衡: %.3f\n', metrics.balance);
    end
    
    if ~isnan(metrics.defender_reward)
        fprintf('防御者奖励: %.3f\n', metrics.defender_reward);
    end
    
    fprintf('迭代耗时: %.2fs\n', iteration_time);
    fprintf('================================\n');
end

function saveIntermediateResults(iteration, config, episode_results)
    %% saveIntermediateResults - 保存中间结果
    
    try
        fprintf('正在保存中间结果 (迭代 %d)...\n', iteration);
        
        % 创建结果目录
        if ~exist('results', 'dir')
            mkdir('results');
        end
        
        % 保存当前状态
        intermediate_data = struct();
        intermediate_data.iteration = iteration;
        intermediate_data.episode_results = episode_results;
        intermediate_data.config = config;
        intermediate_data.timestamp = datestr(now);
        
        filename = sprintf('results/intermediate_iter_%d.mat', iteration);
        save(filename, 'intermediate_data');
        
        fprintf('✓ 中间结果已保存: %s\n', filename);
        
    catch ME
        warning('保存中间结果失败: %s', ME.message);
    end
end

function displayMilestones(iteration, config)
    %% displayMilestones - 显示特殊里程碑
    
    milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    progress = iteration / config.n_iterations;
    
    for milestone = milestones
        if abs(progress - milestone) < (1 / config.n_iterations)
            fprintf('\n🎯 里程碑达成: %.0f%% 完成! (迭代 %d/%d)\n\n', ...
                    milestone * 100, iteration, config.n_iterations);
            break;
        end
    end
end

function checkPerformanceWarnings(iteration, metrics, iteration_time)
    %% checkPerformanceWarnings - 检查性能警告
    
    % 检查迭代时间过长
    if iteration_time > 60  % 超过1分钟
        fprintf('⚠️  警告: 迭代耗时较长 (%.1f秒)\n', iteration_time);
    end
    
    % 检查RADI异常
    if ~isnan(metrics.avg_radi) && (metrics.avg_radi > 10 || metrics.avg_radi < 0.1)
        fprintf('⚠️  警告: RADI指标异常 (%.3f)\n', metrics.avg_radi);
    end
    
    % 检查检测率过低
    if ~isnan(metrics.detection_rate) && metrics.detection_rate < 0.3
        fprintf('⚠️  警告: 检测率较低 (%.1f%%)\n', metrics.detection_rate * 100);
    end
    
    % 检查效率过低
    if ~isnan(metrics.efficiency) && metrics.efficiency < 0.2
        fprintf('⚠️  警告: 资源效率较低 (%.1f%%)\n', metrics.efficiency * 100);
    end
end

function estimateRemainingTime(iteration, config, iteration_time)
    %% estimateRemainingTime - 估算剩余时间
    
    try
        remaining_iterations = config.n_iterations - iteration;
        if remaining_iterations > 0 && iteration_time > 0
            estimated_time = remaining_iterations * iteration_time;
            
            if estimated_time > 3600  % 超过1小时
                hours = floor(estimated_time / 3600);
                minutes = floor((estimated_time - hours * 3600) / 60);
                fprintf('⏰ 预计剩余时间: %d小时%d分钟\n', hours, minutes);
            elseif estimated_time > 60  % 超过1分钟
                minutes = floor(estimated_time / 60);
                seconds = round(estimated_time - minutes * 60);
                fprintf('⏰ 预计剩余时间: %d分钟%d秒\n', minutes, seconds);
            else
                fprintf('⏰ 预计剩余时间: %.0f秒\n', estimated_time);
            end
        end
    catch
        % 时间估算失败，忽略
    end
end

function value = getFieldOrDefault(struct_obj, field_name, default_value)
    %% getFieldOrDefault - 获取结构体字段值或默认值
    if isfield(struct_obj, field_name)
        value = struct_obj.(field_name);
    else
        value = default_value;
    end
end