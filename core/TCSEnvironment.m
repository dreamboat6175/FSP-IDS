classdef TCSEnvironment < handle
    % TCSEnvironment - 增强版战术控制系统仿真环境 (v4.3修复版)
    
    properties (Access = public)
        % 系统参数
        n_stations          % 站点数量
        n_components        % 每个站点的组件数量
        station_values      % 站点价值
        total_resources     % 总资源数
        
        % 状态和动作空间
        state_dim           % 状态空间维度
        action_dim          % 动作空间维度（兼容性）
        action_dim_defender % 防御者动作空间维度
        action_dim_attacker % 攻击者动作空间维度
        
        % 当前状态
        current_state       % 当前环境状态
        time_step           % 时间步
        
        % FSP组件
        attacker_avg_strategy  % 攻击者平均策略
        alpha_ewma             % EWMA学习率
        
        % RADI配置
        radi_config         % RADI计算配置
        radi_score          % 当前RADI分数
        
        % 历史记录
        attack_success_history
        damage_history
        radi_history
        detection_history
        attack_target_history     % 矩阵形式：每行是one-hot编码
        defense_deployment_history % 矩阵形式：每行是部署向量
        
        % 增强的历史记录
        nash_convergence_history
        attack_coverage_history
        defense_effectiveness_history
        strategy_change_history   % [攻击策略变化, 防御策略变化]
        
        % 策略追踪
        prev_attack_strategy
        prev_defense_strategy
        curr_attack_strategy
        curr_defense_strategy
        
        % 配置
        config              % 完整配置
        debug_mode          % 调试模式
        adaptive_alpha      % 自适应alpha
        alpha_min           % 最小alpha
        alpha_max           % 最大alpha
        
        % 新增：初始化控制
        initial_attack_focus    % 初始攻击集中度
        transition_steps        % 过渡步数
    end
    
    properties (Access = private)
        total_components    % 总组件数
        action_templates    % 预定义动作模板
    end
    
    methods
       function obj = TCSEnvironment(config)
            %TCSENVIRONMENT 构造函数（修复版）
            
            if nargin < 1
                error('TCSEnvironment:NoConfig', '需要提供配置参数');
            end
            
            try
                obj.config = config;
                
                % 关键修复：正确的初始化顺序
                % 1. 提取配置参数（包括计算total_components）
                obj.extractConfigParameters(config);
                
                % 2. 验证环境状态
                obj.validateEnvironmentState();
                
                % 3. 初始化环境组件
                obj.initializeEnvironmentComponents();
                
                % 4. 计算空间维度
                obj.calculateSpaceDimensions();
                
                % 5. 重置环境
                obj.reset();
                
                if obj.debug_mode
                    fprintf('[TCSEnvironment v4.3修复版] 初始化完成 - %d站点, %d资源\n', ...
                            obj.n_stations, obj.total_resources);
                end
                
            catch ME
                fprintf('❌ TCSEnvironment 构造失败: %s\n', ME.message);
                fprintf('   错误位置: %s:%d\n', ME.stack(1).file, ME.stack(1).line);
                
                % 提供详细的调试信息
                if exist('obj', 'var')
                    fprintf('   调试信息:\n');
                    if isprop(obj, 'n_stations')
                        fprintf('     n_stations: %d\n', obj.n_stations);
                    end
                    if isprop(obj, 'n_components') && ~isempty(obj.n_components)
                        fprintf('     n_components: %s\n', mat2str(obj.n_components));
                    end
                    if isprop(obj, 'total_components')
                        fprintf('     total_components: %d\n', obj.total_components);
                    end
                end
                
                rethrow(ME);
            end
        end

        
        function state = reset(obj)
            %RESET 重置环境到初始状态
            
            obj.time_step = 0;
            
            % 修复：初始化攻击者策略为集中攻击最高价值站点
            [~, highest_value_station] = max(obj.station_values);
            obj.attacker_avg_strategy = zeros(1, obj.n_stations);
            obj.attacker_avg_strategy(highest_value_station) = obj.initial_attack_focus;
            remaining_prob = 1 - obj.initial_attack_focus;
            other_stations = setdiff(1:obj.n_stations, highest_value_station);
            if ~isempty(other_stations)
                obj.attacker_avg_strategy(other_stations) = remaining_prob / length(other_stations);
            end
            
            obj.radi_score = 0.1; % 初始RADI设为较低值
            
            % 清空历史记录
            obj.clearAllHistoryRecords();
            
            % 重置策略跟踪
            obj.resetStrategyTracking();
            
            % 生成初始状态
            obj.current_state = obj.generateEnvironmentState();
            state = obj.current_state;
            
            if obj.debug_mode
                fprintf('[TCSEnvironment] 环境已重置 - 初始攻击集中于站点%d\n', highest_value_station);
            end
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_deployment, attacker_target)
            %STEP 执行一步环境交互
            
            try
                % 输入验证和修复
                [defender_deployment, attacker_target] = obj.validateAndFixStepInputs(defender_deployment, attacker_target);
                
                % 修复：使用渐进式策略更新
                obj.updateAttackerAverageStrategyProgressive(attacker_target);
                
                % 计算攻击结果
                [attack_success, damage] = obj.computeAttackOutcome(attacker_target, defender_deployment);
                
                % 检测评估
                detection_result = obj.evaluateDetection(attacker_target, defender_deployment, attack_success);
                
                % 计算奖励（使用改进的奖励函数）
                [reward_def, reward_att] = obj.computeRewardsSmooth(attack_success, damage, attacker_target, defender_deployment, detection_result);
                
                % 更新环境状态
                obj.updateEnvironmentState(attack_success, damage, attacker_target, defender_deployment, detection_result);
                
                % 生成下一状态
                next_state = obj.generateEnvironmentState();
                obj.current_state = next_state;
                obj.time_step = obj.time_step + 1;
                
                % 创建信息结构
                info = obj.createStepInfo(attack_success, damage, attacker_target, defender_deployment, detection_result, reward_def, reward_att);
                
            catch ME
                fprintf('[ERROR] TCSEnvironment.step 执行失败: %s\n', ME.message);
                % 返回安全默认值
                next_state = obj.current_state;
                reward_def = 0;
                reward_att = 0;
                info = obj.createDefaultInfo();
            end
        end
        
        function updateAttackerAverageStrategyProgressive(obj, attacker_target)
            %UPDATEATTACKERAVERAGESTRATEGY 渐进式更新攻击者平均策略
            
            target_onehot = zeros(1, obj.n_stations);
            target_onehot(attacker_target) = 1;
            
            % 计算自适应alpha（随时间步增加）
            if obj.time_step < obj.transition_steps
                % 早期阶段：缓慢学习
                current_alpha = obj.alpha_min;
            else
                % 后期阶段：正常学习
                if obj.adaptive_alpha
                    strategy_change = sum(abs(target_onehot - obj.attacker_avg_strategy));
                    if strategy_change > 0.5
                        current_alpha = min(obj.alpha_max, obj.alpha_ewma * 1.2);
                    else
                        current_alpha = max(obj.alpha_min, obj.alpha_ewma * 0.98);
                    end
                else
                    current_alpha = obj.alpha_ewma;
                end
            end
            
            % 更新平均策略
            obj.attacker_avg_strategy = (1 - current_alpha) * obj.attacker_avg_strategy + ...
                                       current_alpha * target_onehot;
            
            % 归一化确保是有效概率分布
            obj.attacker_avg_strategy = max(0.001, obj.attacker_avg_strategy);
            obj.attacker_avg_strategy = obj.attacker_avg_strategy / sum(obj.attacker_avg_strategy);
        end
        
        function [reward_def, reward_att] = computeRewardsSmooth(obj, attack_success, damage, attacker_target, defender_deployment, detection_result)
            %COMPUTEREWARDSSMOOTH 计算平滑的奖励（避免跳变）
            
            % 防御者奖励计算
            % 1. 计算当前RADI
            optimal_deployment = obj.computeOptimalDeploymentSmooth(attacker_target);
            current_radi = obj.calculateRADISmooth(defender_deployment, optimal_deployment);
            obj.radi_score = current_radi;
            
            % 2. RADI奖励（使用平滑函数）
            radi_reward = 2 / (1 + current_radi);  % 范围[1, 2]，避免指数函数的剧烈变化
            
            % 3. 防御成功奖励
            if attack_success
                defense_reward = -0.5;  % 减小惩罚
            else
                defense_reward = 1.0;
            end
            
            % 4. 损害缓解奖励
            max_damage = obj.station_values(attacker_target);
            if max_damage > 0
                damage_mitigation = 1 - (damage / max_damage);
                damage_reward = damage_mitigation * 0.5;
            else
                damage_reward = 0;
            end
            
            % 5. 检测奖励
            if detection_result.detected && ~detection_result.is_false_positive
                detection_reward = 0.3;
            elseif detection_result.is_false_positive
                detection_reward = -0.1;
            else
                detection_reward = 0;
            end
            
            % 6. 资源效率奖励
            resource_usage = sum(defender_deployment) / obj.total_resources;
            efficiency_reward = 0.2 * (1 - abs(resource_usage - 1));
            
            % 7. 学习进度奖励（鼓励早期探索）
            if obj.time_step < obj.transition_steps
                exploration_bonus = 0.2;
            else
                exploration_bonus = 0;
            end
            
            % 综合奖励（调整权重，减小RADI权重）
            reward_def = 0.25 * radi_reward + ...      % 降低RADI权重
                         0.25 * defense_reward + ...    
                         0.20 * damage_reward + ...     
                         0.15 * detection_reward + ...  
                         0.10 * efficiency_reward + ...
                         0.05 * exploration_bonus;
            
            % 添加基础奖励
            reward_def = reward_def + 0.5;
            
            % 限制范围并平滑
            reward_def = max(-1, min(3, reward_def));
            
            % 攻击者奖励
            if attack_success
                reward_att = 2.0 + damage * 0.5;
            else
                reward_att = -0.5;
            end
            
            if detection_result.detected
                reward_att = reward_att - 0.5;
            end
            
            reward_att = max(-2, min(5, reward_att));
        end
        
        function optimal = computeOptimalDeploymentSmooth(obj, attacker_target)
            %COMPUTEOPTIMALDEPLOYMENTSMOOTH 计算平滑的最优部署
            
            optimal = zeros(1, obj.n_stations);
            
            % 方案：基于威胁和价值的混合策略
            % 主要防御被攻击站点，但不过度集中
            main_allocation = 0.4;  % 降低集中度
            optimal(attacker_target) = obj.total_resources * main_allocation;
            
            % 剩余资源基于综合因素分配
            remaining = obj.total_resources * (1 - main_allocation);
            
            % 计算每个站点的防御优先级
            priorities = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                if i == attacker_target
                    priorities(i) = 0;  % 已分配
                else
                    % 基于价值和历史威胁
                    value_factor = obj.station_values(i);
                    threat_factor = obj.attacker_avg_strategy(i);
                    distance_factor = 1 / (1 + abs(i - attacker_target));  % 邻近站点优先级更高
                    
                    priorities(i) = 0.5 * value_factor + 0.3 * threat_factor + 0.2 * distance_factor;
                end
            end
            
            % 分配剩余资源
            if sum(priorities) > 0
                normalized_priorities = priorities / sum(priorities);
                optimal = optimal + normalized_priorities * remaining;
            else
                % 均匀分配剩余资源
                other_stations = setdiff(1:obj.n_stations, attacker_target);
                if ~isempty(other_stations)
                    optimal(other_stations) = remaining / length(other_stations);
                end
            end
            
            % 确保最小分配（避免完全放弃某些站点）
            min_allocation = obj.total_resources * 0.02;
            optimal = max(optimal, min_allocation);
            
            % 归一化
            optimal = optimal * (obj.total_resources / sum(optimal));
        end
        
        function radi = calculateRADISmooth(obj, defender_deployment, optimal_deployment)
            %CALCULATERADISMOOTH 计算平滑的RADI指标
            
            if sum(defender_deployment) == 0 || sum(optimal_deployment) == 0
                radi = 0.5;
                return;
            end
            
            % 归一化部署
            norm_deployment = defender_deployment / sum(defender_deployment);
            norm_optimal = optimal_deployment / sum(optimal_deployment);
            
            % 使用平滑的距离度量
            % 1. 计算加权绝对偏差
            weights = obj.station_values / sum(obj.station_values);
            weighted_deviation = sum(weights .* abs(norm_deployment - norm_optimal));
            
            % 2. 使用sigmoid函数平滑映射到[0,1]
            radi = 2 / (1 + exp(-5 * weighted_deviation)) - 1;
            
            % 3. 添加时间衰减因子（早期RADI较低，鼓励学习）
            if obj.time_step < obj.transition_steps
                time_factor = obj.time_step / obj.transition_steps;
                radi = radi * (0.5 + 0.5 * time_factor);
            end
            
            % 确保在合理范围内
            radi = max(0.01, min(0.99, radi));
        end
        
        function updateEnvironmentState(obj, attack_success, damage, attacker_target, defender_deployment, detection_result)
            %UPDATEENVIRONMENTSTATE 更新环境状态
            try
                % 记录基本历史数据
                obj.attack_success_history(end+1) = double(attack_success);
                obj.damage_history(end+1) = damage;
                
                % 检测结果
                if isstruct(detection_result) && isfield(detection_result, 'detected')
                    obj.detection_history(end+1) = detection_result.detected;
                else
                    obj.detection_history(end+1) = false;
                end
                
                % 记录攻击目标（one-hot编码）
                target_vector = zeros(1, obj.n_stations);
                if attacker_target >= 1 && attacker_target <= obj.n_stations
                    target_vector(attacker_target) = 1;
                end
                
                % 更新矩阵形式历史记录
                if isempty(obj.attack_target_history)
                    obj.attack_target_history = target_vector;
                    obj.defense_deployment_history = defender_deployment(:)';
                else
                    obj.attack_target_history(end+1, :) = target_vector;
                    obj.defense_deployment_history(end+1, :) = defender_deployment(:)';
                end
                
                % 计算并更新RADI
                current_radi = obj.calculateRADI(defender_deployment);
                obj.radi_history(end+1) = current_radi;
                obj.radi_score = current_radi;
                
                % 更新增强指标
                obj.updateEnhancedMetrics(attack_success, damage, attacker_target, defender_deployment, detection_result);
                
            catch ME
                warning('TCSEnvironment:UpdateError', '更新环境状态时出错: %s', ME.message);
            end
        end
        
        function optimal_allocation = computeOptimalAllocation(obj, varargin)
            %COMPUTEOPTIMALALLOCATION 计算最优资源分配
            try
                % 初始均匀分配
                optimal_allocation = ones(1, obj.n_stations) / obj.n_stations * obj.total_resources;
                
                % 基于历史攻击数据优化分配
                if ~isempty(obj.attack_target_history) && size(obj.attack_target_history, 1) > 0
                    attack_frequencies = sum(obj.attack_target_history, 1);
                    if sum(attack_frequencies) > 0
                        attack_probabilities = attack_frequencies / sum(attack_frequencies);
                        optimal_allocation = attack_probabilities * obj.total_resources;
                    end
                end
                
                % 基于RADI配置的最优分配
                if isfield(obj.radi_config, 'optimal_allocation') && ...
                   length(obj.radi_config.optimal_allocation) == obj.n_stations
                    radi_optimal = obj.radi_config.optimal_allocation(:)';
                    if sum(radi_optimal) > 0
                        radi_optimal = radi_optimal / sum(radi_optimal);
                        optimal_allocation = radi_optimal * obj.total_resources;
                    end
                end
                
                % 确保分配有效性
                optimal_allocation = max(0, optimal_allocation);
                if sum(optimal_allocation) > obj.total_resources
                    optimal_allocation = optimal_allocation / sum(optimal_allocation) * obj.total_resources;
                end
                
                optimal_allocation = optimal_allocation(:)';
                
            catch ME
                warning('TCSEnvironment:ComputeOptimalError', '计算最优分配时出错: %s', ME.message);
                optimal_allocation = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            end
        end
        
        %% ========== 策略管理方法 ==========
        
        function updateStrategies(obj, attack_strategy, defense_strategy)
            % 更新策略并记录变化
            if length(attack_strategy) ~= obj.n_stations || length(defense_strategy) ~= obj.n_stations
                warning('TCSEnvironment:InvalidStrategy', '策略向量长度与站点数不匹配');
                return;
            end
            
            obj.prev_attack_strategy = obj.curr_attack_strategy;
            obj.prev_defense_strategy = obj.curr_defense_strategy;
            
            obj.curr_attack_strategy = attack_strategy(:)';
            obj.curr_defense_strategy = defense_strategy(:)';
            
            % 归一化策略
            if sum(obj.curr_attack_strategy) > 0
                obj.curr_attack_strategy = obj.curr_attack_strategy / sum(obj.curr_attack_strategy);
            end
            if sum(obj.curr_defense_strategy) > 0
                obj.curr_defense_strategy = obj.curr_defense_strategy / sum(obj.curr_defense_strategy);
            end
        end
        
        %% ========== 信息获取方法 ==========
        
        function info = getEnvironmentInfo(obj)
            %GETENVIRONMENTINFO 获取环境信息
            
            info = struct();
            info.n_stations = obj.n_stations;
            info.total_resources = obj.total_resources;
            info.time_step = obj.time_step;
            info.current_radi = obj.radi_score;
            
            % 性能统计
            info.recent_success_rate = obj.computeRecentSuccessRate();
            info.recent_detection_rate = obj.computeRecentDetectionRate();
            info.recent_radi = obj.computeRecentRADI();
            
            % 增强指标统计
            if ~isempty(obj.nash_convergence_history)
                info.recent_nash_convergence = mean(obj.nash_convergence_history(max(1, end-19):end));
            else
                info.recent_nash_convergence = 1.0;
            end
            
            if ~isempty(obj.attack_coverage_history)
                info.recent_attack_coverage = mean(obj.attack_coverage_history(max(1, end-19):end));
            else
                info.recent_attack_coverage = 0.5;
            end
            
            if ~isempty(obj.defense_effectiveness_history)
                info.recent_defense_effectiveness = mean(obj.defense_effectiveness_history(max(1, end-19):end));
            else
                info.recent_defense_effectiveness = 0.5;
            end
        end
        
        function summary = getDataSummary(obj)
            % 获取数据摘要（用于调试和监控）
            
            summary = struct();
            summary.total_steps = obj.time_step;
            
            % 原有指标
            summary.radi_data_points = length(obj.radi_history);
            summary.attack_success_data_points = length(obj.attack_success_history);
            
            % 新增指标
            summary.nash_convergence_data_points = length(obj.nash_convergence_history);
            summary.attack_coverage_data_points = length(obj.attack_coverage_history);
            summary.defense_effectiveness_data_points = length(obj.defense_effectiveness_history);
            summary.strategy_change_data_points = size(obj.strategy_change_history, 1);
            
            % 最终值
            if ~isempty(obj.radi_history)
                summary.final_radi = obj.radi_history(end);
            else
                summary.final_radi = NaN;
            end
            
            if ~isempty(obj.nash_convergence_history)
                summary.final_nash_convergence = obj.nash_convergence_history(end);
            else
                summary.final_nash_convergence = NaN;
            end
            
            if ~isempty(obj.attack_coverage_history)
                summary.final_attack_coverage = obj.attack_coverage_history(end);
            else
                summary.final_attack_coverage = NaN;
            end
        end
        
        %% ========== 兼容性方法（与AgentFactory配合） ==========
        
        function deployment = parseDefenderAction(obj, action)
            %PARSEDEFENDERACTION 解析防御者动作
            
            if isscalar(action)
                % 使用预定义模板
                template_idx = min(max(round(action), 1), size(obj.action_templates, 1));
                deployment = obj.action_templates(template_idx, :) * obj.total_resources;
            elseif length(action) == obj.n_stations
                deployment = action;
                if sum(deployment) > 0
                    deployment = deployment / sum(deployment) * obj.total_resources;
                end
            else
                deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            end
            
            deployment = max(0, deployment);
        end
        
        function target = parseAttackerAction(obj, action)
            %PARSEATTACKERACTION 解析攻击者动作
            
            if isscalar(action)
                target = min(max(round(action), 1), obj.n_stations);
            else
                [~, target] = max(action);
            end
        end
        function radi = calculateRADI(obj, defender_deployment, optimal_deployment)
            %CALCULATERADI 计算RADI指标
            % 输入:
            %   defender_deployment - 当前防御部署向量
            %   optimal_deployment - 最优防御部署向量（可选）
            % 输出:
            %   radi - RADI指标值
            
            try
                % 参数验证
                if nargin < 2 || isempty(defender_deployment)
                    radi = 1.0;
                    return;
                end
                
                % 如果没有提供最优部署，计算基于价值的最优部署
                if nargin < 3 || isempty(optimal_deployment)
                    if isfield(obj.radi_config, 'optimal_allocation') && ...
                       length(obj.radi_config.optimal_allocation) == obj.n_stations
                        optimal_deployment = obj.radi_config.optimal_allocation * obj.total_resources;
                    else
                        % 基于站点价值的加权分配
                        if sum(obj.station_values) > 0
                            weights = obj.station_values / sum(obj.station_values);
                            optimal_deployment = weights * obj.total_resources;
                        else
                            % 均匀分配作为备选
                            optimal_deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
                        end
                    end
                end
                
                % 调用外部函数计算RADI
                if exist('calculateRADI', 'file') == 2
                    % 使用外部utils/calculateRADI.m函数
                    radi = calculateRADI(defender_deployment, optimal_deployment, obj.radi_config);
                else
                    % 备用内部计算
                    radi = obj.calculateRADIInternal(defender_deployment, optimal_deployment);
                end
                
            catch ME
                warning('calculateRADI计算出错: %s', ME.message);
                radi = 1.0; % 返回默认中等偏差值
            end
        end
        
        function radi = calculateRADIInternal(obj, defender_deployment, optimal_deployment)
            %CALCULATERADIINTERNAL 内部RADI计算方法（备用）
            
            % 确保输入有效
            if sum(defender_deployment) == 0
                radi = 1.0;
                return;
            end
            
            % 归一化部署
            normalized_deployment = defender_deployment / sum(defender_deployment);
            normalized_optimal = optimal_deployment / sum(optimal_deployment);
            
            % 获取权重
            if isfield(obj.radi_config, 'weights') && ...
               length(obj.radi_config.weights) == obj.n_stations
                weights = obj.radi_config.weights;
            else
                % 基于站点价值的权重
                if sum(obj.station_values) > 0
                    weights = obj.station_values / sum(obj.station_values);
                else
                    weights = ones(1, obj.n_stations) / obj.n_stations;
                end
            end
            
            % 计算加权偏差
            deviation = abs(normalized_deployment - normalized_optimal);
            radi = sum(weights .* deviation);
            
            % 限制在合理范围内
            radi = max(0, min(radi, 2));
        end
    end
    
    %% ========== 私有方法 ==========
    methods (Access = private)
        
      function extractConfigParameters(obj, config)
            %EXTRACTCONFIGPARAMETERS 从配置中提取参数（修复版）
            
            try
                % 系统参数
                if isfield(config, 'system') && isstruct(config.system)
                    % 必须参数验证
                    if ~isfield(config.system, 'n_stations') || config.system.n_stations <= 0
                        error('TCSEnvironment:InvalidConfig', 'config.system.n_stations 必须是正整数');
                    end
                    obj.n_stations = config.system.n_stations;
                    obj.total_resources = config.system.total_resources;
                    
                    % 关键修复：配置字段名修正
                    if isfield(config.system, 'n_components_per_station')
                        obj.n_components = config.system.n_components_per_station(:)'; % 确保行向量
                        
                        % 验证组件数组长度
                        if length(obj.n_components) ~= obj.n_stations
                            error('TCSEnvironment:ComponentMismatch', ...
                                  'n_components_per_station 长度 (%d) 与 n_stations (%d) 不匹配', ...
                                  length(obj.n_components), obj.n_stations);
                        end
                        
                    elseif isfield(config.system, 'n_components') % 备用字段名
                        obj.n_components = config.system.n_components(:)';
                        if length(obj.n_components) ~= obj.n_stations
                            error('TCSEnvironment:ComponentMismatch', ...
                                  'n_components 长度与 n_stations 不匹配');
                        end
                    else
                        % 生成默认组件数量
                        obj.n_components = randi([3, 7], 1, obj.n_stations);
                        warning('TCSEnvironment:UsingDefaultComponents', ...
                                '使用默认组件配置，每站点3-7个随机组件');
                    end
                    
                    % 关键修复：在这里计算 total_components
                    obj.total_components = sum(obj.n_components);
                    
                    % 验证 total_components
                    if obj.total_components <= 0
                        error('TCSEnvironment:InvalidTotalComponents', 'total_components 必须大于0');
                    end
                    
                    % 获取或生成站点价值
                    if isfield(config.system, 'station_values') && ...
                       length(config.system.station_values) == obj.n_stations
                        obj.station_values = config.system.station_values(:)';
                    else
                        % 在 total_components 已设置后调用
                        obj.generateStationValues();
                    end
                    
                elseif isfield(config, 'n_stations') % 兼容旧格式
                    obj.n_stations = config.n_stations;
                    obj.n_components = config.n_components_per_station;
                    obj.total_resources = config.total_resources;
                    
                    % 同样的验证
                    if length(obj.n_components) ~= obj.n_stations
                        error('TCSEnvironment:ComponentMismatch', ...
                              'n_components_per_station 长度与 n_stations 不匹配');
                    end
                    
                    obj.total_components = sum(obj.n_components);
                    obj.generateStationValues();
                    
                else
                    % 使用一致的默认值
                    obj.n_stations = 5; % 改为5个站点，避免过大
                    obj.n_components = randi([3, 7], 1, obj.n_stations);
                    obj.total_resources = 100;
                    obj.total_components = sum(obj.n_components);
                    obj.generateStationValues();
                    
                    warning('TCSEnvironment:UsingAllDefaults', '使用所有默认系统参数');
                end
                
                % RADI配置
                if isfield(config, 'radi') && isstruct(config.radi)
                    obj.radi_config = config.radi;
                else
                    obj.radi_config = obj.createDefaultRADIConfig();
                end
                
                % FSP学习参数
                obj.alpha_ewma = obj.getConfigValue(config, {'simulation', 'alpha_ewma'}, 0.1);
                obj.adaptive_alpha = obj.getConfigValue(config, {'simulation', 'adaptive_alpha'}, true);
                obj.alpha_min = obj.getConfigValue(config, {'simulation', 'alpha_min'}, 0.05);
                obj.alpha_max = obj.getConfigValue(config, {'simulation', 'alpha_max'}, 0.3);
                
                % 检测系统参数初始化
                obj.detection_enabled = obj.getConfigValue(config, {'detection', 'enabled'}, false);
                obj.base_detection_rate = obj.getConfigValue(config, {'detection', 'base_rate'}, 0.1);
                obj.detection_sensitivity = obj.getConfigValue(config, {'detection', 'sensitivity'}, 0.3);
                obj.false_positive_rate = obj.getConfigValue(config, {'detection', 'false_positive_rate'}, 0.05);
                
                % 调试模式
                obj.debug_mode = obj.getConfigValue(config, {'debug', 'debug_mode'}, false);
                
                % 新增：初始化控制参数
                obj.initial_attack_focus = 0.7;
                obj.transition_steps = 100;
                
                % 打印初始化信息
                if obj.debug_mode
                    fprintf('[TCSEnvironment] 参数提取完成:\n');
                    fprintf('  站点数: %d\n', obj.n_stations);
                    fprintf('  组件数: %s\n', mat2str(obj.n_components));
                    fprintf('  总组件数: %d\n', obj.total_components);
                    fprintf('  总资源: %d\n', obj.total_resources);
                end
                
            catch ME
                fprintf('❌ extractConfigParameters 失败: %s\n', ME.message);
                fprintf('   错误位置: %s:%d\n', ME.stack(1).file, ME.stack(1).line);
                rethrow(ME);
            end
        end

        function value = getConfigValue(obj, config, field_path, default_value)
            %GETCONFIGVALUE 安全获取配置值的辅助方法
            % 输入:
            %   config - 配置结构体
            %   field_path - 字段路径（cell数组或字符串）
            %   default_value - 默认值
            % 输出:
            %   value - 配置值或默认值
            
            try
                if ischar(field_path)
                    field_path = {field_path};
                end
                
                current = config;
                for i = 1:length(field_path)
                    field = field_path{i};
                    if isfield(current, field)
                        current = current.(field);
                    else
                        value = default_value;
                        return;
                    end
                end
                
                value = current;
                
            catch
                value = default_value;
            end
        end
        
        function radi_config = createDefaultRADIConfig(obj)
            %CREATEDEFAULTRADICONFIG 创建默认RADI配置
            
            radi_config = struct();
            radi_config.method = 'weighted_deviation';
            radi_config.weights = ones(1, obj.n_stations) / obj.n_stations; % 均匀权重
            radi_config.normalization = true;
            radi_config.range = [0, 2];
            
            % 基于站点价值的权重（如果已经计算）
            if ~isempty(obj.station_values) && sum(obj.station_values) > 0
                radi_config.weights = obj.station_values / sum(obj.station_values);
            end
        end

        function initializeEnvironmentComponents(obj)
            %INITIALIZEENVIRONMENTCOMPONENTS 初始化环境组件
            
            obj.total_components = sum(obj.n_components);
            obj.action_templates = obj.createActionTemplates();
            
            % 初始化历史记录
            obj.clearAllHistoryRecords();
            
            % 初始化策略跟踪
            obj.resetStrategyTracking();
        end
        
        function calculateSpaceDimensions(obj)
            %CALCULATESPACEDIMENSIONS 计算状态和动作空间维度
            
            % 状态维度：[攻击者平均策略, 时间归一化, RADI]
            obj.state_dim = obj.n_stations + 2;
            
            % 动作维度
            obj.action_dim = obj.n_stations; % 兼容性属性
            obj.action_dim_defender = obj.n_stations;
            obj.action_dim_attacker = obj.n_stations;
        end
        
        function generateStationValues(obj)
            %GENERATESTATIONVALUES 生成站点价值（安全版本）
            
            try
                % 输入验证
                if ~isprop(obj, 'n_stations') || obj.n_stations <= 0
                    error('TCSEnvironment:InvalidNStations', 'n_stations 必须已初始化且大于0');
                end
                
                if ~isprop(obj, 'n_components') || isempty(obj.n_components)
                    error('TCSEnvironment:InvalidComponents', 'n_components 必须已初始化');
                end
                
                if length(obj.n_components) ~= obj.n_stations
                    error('TCSEnvironment:ComponentStationMismatch', ...
                          'n_components 长度 (%d) 与 n_stations (%d) 不匹配', ...
                          length(obj.n_components), obj.n_stations);
                end
                
                if ~isprop(obj, 'total_components') || obj.total_components <= 0
                    error('TCSEnvironment:InvalidTotalComponents', 'total_components 必须已正确设置');
                end
                
                % 验证一致性
                expected_total = sum(obj.n_components);
                if obj.total_components ~= expected_total
                    error('TCSEnvironment:TotalComponentsMismatch', ...
                          'total_components (%d) 与 sum(n_components) (%d) 不一致', ...
                          obj.total_components, expected_total);
                end
                
                % 安全生成组件重要性
                component_importance = rand(1, obj.total_components);
                obj.station_values = zeros(1, obj.n_stations);
                
                % 安全的索引计算
                idx = 1;
                for i = 1:obj.n_stations
                    n_comp = obj.n_components(i);
                    
                    % 关键边界检查
                    end_idx = idx + n_comp - 1;
                    if end_idx > obj.total_components
                        error('TCSEnvironment:IndexOutOfBounds', ...
                              '站点 %d: 索引范围 [%d:%d] 超出 component_importance 长度 (%d)', ...
                              i, idx, end_idx, obj.total_components);
                    end
                    
                    % 安全计算站点价值
                    obj.station_values(i) = sum(component_importance(idx:end_idx));
                    idx = idx + n_comp;
                end
                
                % 归一化处理
                if sum(obj.station_values) == 0
                    % 如果所有值都是0，使用均匀分布
                    obj.station_values = ones(1, obj.n_stations) / obj.n_stations;
                    warning('TCSEnvironment:ZeroStationValues', '所有站点价值为0，使用均匀分布');
                else
                    % 正常归一化
                    obj.station_values = obj.station_values / sum(obj.station_values);
                    obj.station_values = obj.station_values .^ 0.8; % 减少极端差异
                    obj.station_values = obj.station_values / sum(obj.station_values);
                end
                
                % 最终验证
                if any(isnan(obj.station_values)) || any(isinf(obj.station_values))
                    error('TCSEnvironment:InvalidStationValues', '生成的站点价值包含 NaN 或 Inf');
                end
                
                if abs(sum(obj.station_values) - 1.0) > 1e-10
                    warning('TCSEnvironment:NormalizationWarning', '站点价值归一化精度问题');
                end
                
            catch ME
                fprintf('❌ generateStationValues 失败: %s\n', ME.message);
                fprintf('   当前状态: n_stations=%d, total_components=%d\n', ...
                        obj.n_stations, obj.total_components);
                if ~isempty(obj.n_components)
                    fprintf('   n_components=%s\n', mat2str(obj.n_components));
                end
                rethrow(ME);
            end
        end

        
        function templates = createActionTemplates(obj)
            %CREATEACTIONTEMPLATES 创建动作模板
            
            n_templates = 5;
            templates = zeros(n_templates, obj.n_stations);
            
            % 均匀分配
            templates(1, :) = ones(1, obj.n_stations) / obj.n_stations;
            
            % 基于价值的分配
            templates(2, :) = obj.station_values / sum(obj.station_values);
            
            % 随机分配
            for i = 3:n_templates
                random_weights = rand(1, obj.n_stations);
                templates(i, :) = random_weights / sum(random_weights);
            end
        end
        
        function clearAllHistoryRecords(obj)
            %CLEARALLHISTORYRECORDS 清空所有历史记录
            
            % 基本历史记录
            obj.attack_success_history = [];
            obj.damage_history = [];
            obj.radi_history = [];
            obj.detection_history = [];
            
            % 矩阵形式历史记录（初始化为0行N列）
            obj.attack_target_history = zeros(0, obj.n_stations);
            obj.defense_deployment_history = zeros(0, obj.n_stations);
            
            % 增强指标历史
            obj.nash_convergence_history = [];
            obj.attack_coverage_history = [];
            obj.defense_effectiveness_history = [];
            obj.strategy_change_history = zeros(0, 2);
        end
        
        function resetStrategyTracking(obj)
            %RESETSTRATEGYTRACKING 重置策略跟踪
            
            obj.prev_attack_strategy = [];
            obj.prev_defense_strategy = [];
            obj.curr_attack_strategy = [];
            obj.curr_defense_strategy = [];
        end
        
        function [defender_deployment, attacker_target] = validateAndFixStepInputs(obj, defender_deployment, attacker_target)
            %VALIDATEANDFIXSTEPINPUTS 验证和修复输入维度
            
            % 修复防御者部署向量
            if isempty(defender_deployment)
                defender_deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            end
            
            defender_deployment = defender_deployment(:)';
            
            if length(defender_deployment) ~= obj.n_stations
                if length(defender_deployment) < obj.n_stations
                    padding = ones(1, obj.n_stations - length(defender_deployment)) * (obj.total_resources / obj.n_stations / 10);
                    defender_deployment = [defender_deployment, padding];
                else
                    defender_deployment = defender_deployment(1:obj.n_stations);
                end
            end
            
            defender_deployment = max(0, defender_deployment);
            
            if sum(defender_deployment) > obj.total_resources * 1.1
                defender_deployment = defender_deployment * (obj.total_resources / sum(defender_deployment));
            end
            
            % 修复攻击者目标
            if isempty(attacker_target)
                attacker_target = randi(obj.n_stations);
            end
            
            if isscalar(attacker_target)
                attacker_target = max(1, min(obj.n_stations, round(attacker_target)));
            else
                if length(attacker_target) ~= obj.n_stations
                    if length(attacker_target) < obj.n_stations
                        attacker_target = [attacker_target, zeros(1, obj.n_stations - length(attacker_target))];
                    else
                        attacker_target = attacker_target(1:obj.n_stations);
                    end
                end
                [~, attacker_target] = max(attacker_target);
            end
        end
        
        %% ========== 核心计算方法 ==========
        
        function updateAttackerAverageStrategy(obj, attacker_target)
            %UPDATEATTACKERAVERAGESTRATEGY 更新攻击者平均策略（增强版FSP）
            
            target_onehot = zeros(1, obj.n_stations);
            target_onehot(attacker_target) = 1;
            
            % 自适应alpha机制
            if obj.adaptive_alpha
                % 计算策略变化程度
                strategy_change = sum(abs(target_onehot - obj.attacker_avg_strategy));
                
                % 如果变化大，提高alpha以快速适应
                if strategy_change > 0.5
                    current_alpha = min(obj.alpha_max, obj.alpha_ewma * 1.5);
                else
                    current_alpha = max(obj.alpha_min, obj.alpha_ewma * 0.95);
                end
            else
                current_alpha = obj.alpha_ewma;
            end
            
            % 更新平均策略
            obj.attacker_avg_strategy = (1 - current_alpha) * obj.attacker_avg_strategy + ...
                                       current_alpha * target_onehot;
            
            % 添加少量噪声防止过度确定性
            noise = randn(1, obj.n_stations) * 0.01;
            obj.attacker_avg_strategy = obj.attacker_avg_strategy + noise;
            
            % 归一化确保是有效概率分布
            obj.attacker_avg_strategy = max(0, obj.attacker_avg_strategy);
            obj.attacker_avg_strategy = obj.attacker_avg_strategy / sum(obj.attacker_avg_strategy);
        end      
        
        function [attack_success, damage] = computeAttackOutcome(obj, attacker_target, defender_deployment)
            %COMPUTEATTACKOUTCOME 计算攻击结果
            
            target_defense = defender_deployment(attacker_target);
            target_value = obj.station_values(attacker_target);
            
            % 攻击成功概率
            if obj.total_resources > 0
                defense_ratio = target_defense / obj.total_resources;
            else
                defense_ratio = 0;
            end
            
            attack_success_prob = max(0.1, min(0.9, 1 - defense_ratio));
            attack_success = rand() < attack_success_prob;
            
            % 计算损害
            if attack_success
                base_damage = target_value;
                defense_mitigation = defense_ratio * 0.5;
                damage = base_damage * (1 - defense_mitigation);
            else
                damage = 0;
            end
            
            damage = max(0, damage);
        end
        
        function detection_result = evaluateDetection(obj, attacker_target, defender_deployment, attack_success)
            %EVALUATEDETECTION 评估检测结果（修复版）
            
            detection_result = struct();
            
            % 确保检测系统参数已初始化
            if ~isprop(obj, 'detection_enabled') || ~obj.detection_enabled
                detection_result.detected = false;
                detection_result.detection_prob = 0;
                detection_result.is_false_positive = false;
                return;
            end
            
            % 计算检测概率
            target_defense = defender_deployment(attacker_target);
            if obj.total_resources > 0
                defense_factor = target_defense / obj.total_resources;
            else
                defense_factor = 0;
            end
            
            detection_prob = obj.base_detection_rate + defense_factor * obj.detection_sensitivity;
            detection_prob = min(0.95, max(0.05, detection_prob)); % 限制范围
            
            % 确定检测结果
            if attack_success
                detected = rand() < detection_prob;
                is_false_positive = false;
            else
                detected = rand() < obj.false_positive_rate;
                is_false_positive = detected;
            end
            
            detection_result.detected = detected;
            detection_result.detection_prob = detection_prob;
            detection_result.is_false_positive = is_false_positive;
        end

        function validateEnvironmentState(obj)
            %VALIDATEENVIRONMENTSTATE 验证环境状态的一致性
            
            errors = {};
            
            % 检查基本参数
            if obj.n_stations <= 0
                errors{end+1} = 'n_stations 必须大于0';
            end
            
            if length(obj.n_components) ~= obj.n_stations
                errors{end+1} = sprintf('n_components 长度 (%d) 与 n_stations (%d) 不匹配', ...
                                       length(obj.n_components), obj.n_stations);
            end
            
            if obj.total_components ~= sum(obj.n_components)
                errors{end+1} = sprintf('total_components (%d) 与 sum(n_components) (%d) 不一致', ...
                                       obj.total_components, sum(obj.n_components));
            end
            
            if length(obj.station_values) ~= obj.n_stations
                errors{end+1} = sprintf('station_values 长度 (%d) 与 n_stations (%d) 不匹配', ...
                                       length(obj.station_values), obj.n_stations);
            end
            
            if abs(sum(obj.station_values) - 1.0) > 1e-6
                errors{end+1} = sprintf('station_values 归一化错误，总和为 %.6f', sum(obj.station_values));
            end
            
            % 输出结果
            if isempty(errors)
                if obj.debug_mode
                    fprintf('✅ 环境状态验证通过\n');
                end
            else
                fprintf('❌ 环境状态验证失败:\n');
                for i = 1:length(errors)
                    fprintf('  %d. %s\n', i, errors{i});
                end
                error('TCSEnvironment:ValidationFailed', '环境状态验证失败');
            end
        end

        
        function [reward_def, reward_att] = computeRewards(obj, attack_success, damage, ...
    attacker_target, defender_deployment, detection_result)
            %COMPUTEREWARDS 计算增强版奖励函数
            
            %% 攻击者奖励（保持原有逻辑）
            if attack_success
                reward_att = damage;
                if detection_result.detected && ~detection_result.is_false_positive
                    reward_att = reward_att * 0.5;
                end
            else
                reward_att = -0.1;
            end
            
            %% 防御者奖励（全新设计）
            
            % 1. 基础奖励（确保有正向激励）
            base_reward = 1.0;
            
            % 2. 损害惩罚/奖励
            if attack_success
                damage_penalty = -damage * 2;  % 失败时的惩罚
            else
                damage_reward = 2.0;  % 成功防御的奖励
            end
            
            % 3. RADI性能奖励（核心改进）
            % 先计算当前RADI
            optimal_deployment = obj.computeOptimalDeployment(attacker_target);
            current_radi = obj.calculateRADI(defender_deployment, optimal_deployment);
            obj.radi_score = current_radi;  % 保存供其他组件使用
            
            % 使用指数函数将RADI转换为奖励（RADI越小奖励越高）
            radi_reward = 2 * exp(-current_radi * 3);  % 范围约[0, 2]
            
            % 4. 检测奖励
            detection_reward = 0;
            if detection_result.detected && ~detection_result.is_false_positive
                detection_reward = 0.5;  % 正确检测
            elseif detection_result.is_false_positive
                detection_reward = -0.1;  % 误报惩罚（减小）
            end
            
            % 5. 资源效率奖励
            used_resources = sum(defender_deployment);
            if used_resources > 0 && obj.total_resources > 0
                efficiency = 1 - abs(used_resources - obj.total_resources) / obj.total_resources;
                efficiency_reward = efficiency * 0.3;
            else
                efficiency_reward = 0;
            end
            
            % 6. 防御平衡奖励（避免资源过度集中）
            if length(defender_deployment) > 1
                deployment_std = std(defender_deployment / sum(defender_deployment));
                balance_reward = (1 - min(deployment_std, 1)) * 0.2;
            else
                balance_reward = 0;
            end
            
            % 综合计算（使用明确的权重）
            wradi = 0.4;    % RADI权重
            wdamage = 0.3;  % 损害权重
            wdetect = 0.15; % 检测权重
            weffic = 0.1;   % 效率权重
            wbalance = 0.05; % 平衡权重
            
            if attack_success
                reward_def = base_reward + ...
                            wradi * radi_reward + ...
                            wdamage * damage_penalty + ...
                            wdetect * detection_reward + ...
                            weffic * efficiency_reward + ...
                            wbalance * balance_reward;
            else
                reward_def = base_reward + ...
                            wradi * radi_reward + ...
                            wdamage * damage_reward + ...
                            wdetect * detection_reward + ...
                            weffic * efficiency_reward + ...
                            wbalance * balance_reward;
            end
            
            % 确保奖励在合理范围内
            reward_def = max(-2, min(5, reward_def));
            
            % 可选：输出调试信息
            if obj.debug_mode
                fprintf('[Reward] RADI: %.3f, RADI_reward: %.3f, Total: %.3f\n', ...
                        current_radi, radi_reward, reward_def);
            end
        end

        function optimal = computeOptimalDeployment(obj, attacker_target)
            %COMPUTEOPTIMALDEPLOYMENT 计算事后最优部署（知道攻击目标后的最优分配）
            
            optimal = zeros(1, obj.n_stations);
            
            % 方案1：集中防御策略（60%资源给被攻击站点）
            main_allocation = 0.6;
            optimal(attacker_target) = obj.total_resources * main_allocation;
            
            % 剩余40%资源基于站点价值和历史威胁分配
            remaining = obj.total_resources * (1 - main_allocation);
            other_stations = setdiff(1:obj.n_stations, attacker_target);
            
            if ~isempty(other_stations)
                % 结合站点价值和威胁感知
                values = obj.station_values(other_stations);
                threats = obj.attacker_avg_strategy(other_stations);
                
                % 综合权重：70%基于价值，30%基于威胁
                combined_weights = 0.7 * values + 0.3 * threats;
                
                if sum(combined_weights) > 0
                    normalized_weights = combined_weights / sum(combined_weights);
                    optimal(other_stations) = normalized_weights * remaining;
                else
                    % 均匀分配剩余资源
                    optimal(other_stations) = remaining / length(other_stations);
                end
            end
            
            % 确保资源总和正确
            optimal = optimal * (obj.total_resources / sum(optimal));
        end
        
        function success_rate = computeRecentSuccessRate(obj)
            %COMPUTERECENTSUCCESSRATE 计算最近攻击成功率
            
            if isempty(obj.attack_success_history)
                success_rate = 0;
                return;
            end
            
            recent_window = min(50, length(obj.attack_success_history));
            recent_data = obj.attack_success_history(end-recent_window+1:end);
            success_rate = mean(recent_data);
        end
        
        function detection_rate = computeRecentDetectionRate(obj)
            %COMPUTERECENTDETECTIONRATE 计算最近检测率
            
            if isempty(obj.detection_history)
                detection_rate = 0;
                return;
            end
            
            recent_window = min(50, length(obj.detection_history));
            recent_data = obj.detection_history(end-recent_window+1:end);
            detection_rate = mean(recent_data);
        end
        
        function recent_radi = computeRecentRADI(obj)
            %COMPUTERECENTRADI 计算最近RADI
            
            if isempty(obj.radi_history)
                recent_radi = 0.5;
                return;
            end
            
            recent_window = min(20, length(obj.radi_history));
            recent_data = obj.radi_history(end-recent_window+1:end);
            recent_radi = mean(recent_data);
        end
        
        function state = generateEnvironmentState(obj)
            %GENERATEENVIRONMENTSTATE 生成增强的环境状态向量
            
            % 1. 攻击者平均策略（n维）
            attacker_avg = obj.attacker_avg_strategy;
            
            % 2. 最近k次攻击历史的频率分布（n维）
            recent_attack_freq = zeros(1, obj.n_stations);
            
            % 从attack_target_history矩阵中提取攻击目标
            if ~isempty(obj.attack_target_history)
                recent_k = min(10, size(obj.attack_target_history, 1));
                if recent_k > 0
                    % attack_target_history是one-hot编码矩阵，每行只有一个1
                    recent_attacks_matrix = obj.attack_target_history(end-recent_k+1:end, :);
                    % 计算每个站点被攻击的频率
                    recent_attack_freq = sum(recent_attacks_matrix, 1) / recent_k;
                end
            end
            
            % 3. 站点价值（n维）
            normalized_values = obj.station_values / sum(obj.station_values);
            
            % 4. 最近防御部署（n维）
            if ~isempty(obj.defense_deployment_history)
                recent_deployment = obj.defense_deployment_history(end, :);
                recent_deployment = recent_deployment / sum(recent_deployment);
            else
                recent_deployment = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 5. 性能指标（3维）
            time_norm = min(obj.time_step / 1000, 1.0);
            recent_radi = obj.computeRecentRADI();
            recent_success_rate = obj.computeRecentSuccessRate();
            
            % 组合状态向量
            state = [attacker_avg, recent_attack_freq, normalized_values, ...
                     recent_deployment, time_norm, recent_radi, recent_success_rate];
            
            % 更新状态维度
            obj.state_dim = length(state);
        end
        
        function updateEnhancedMetrics(obj, attack_success, damage, attacker_target, defender_deployment, detection_result)
            %UPDATEENHANCEDMETRICS 更新增强指标
            
            % Nash均衡收敛度
            nash_conv = obj.calculateNashConvergence();
            obj.nash_convergence_history(end+1) = nash_conv;
            
            % 攻击覆盖率
            coverage = obj.calculateAttackCoverage(defender_deployment, attack_success, detection_result);
            obj.attack_coverage_history(end+1) = coverage;
            
            % 防御有效性
            effectiveness = obj.calculateDefenseEffectiveness(defender_deployment, damage, attack_success);
            obj.defense_effectiveness_history(end+1) = effectiveness;
            
            % 策略变化
            if ~isempty(obj.prev_attack_strategy) && ~isempty(obj.prev_defense_strategy) && ...
               ~isempty(obj.curr_attack_strategy) && ~isempty(obj.curr_defense_strategy)
                attack_change = norm(obj.curr_attack_strategy - obj.prev_attack_strategy, 2);
                defense_change = norm(obj.curr_defense_strategy - obj.prev_defense_strategy, 2);
                obj.strategy_change_history(end+1, :) = [attack_change, defense_change];
            else
                obj.strategy_change_history(end+1, :) = [0, 0];
            end
        end
        
        function nash_convergence = calculateNashConvergence(obj)
            %CALCULATENASHCONVERGENCE 计算Nash均衡收敛度
            
            if isempty(obj.prev_attack_strategy) || isempty(obj.prev_defense_strategy) || ...
               isempty(obj.curr_attack_strategy) || isempty(obj.curr_defense_strategy)
                nash_convergence = 1.0;
                return;
            end
            
            attack_change = norm(obj.curr_attack_strategy - obj.prev_attack_strategy, 2);
            defense_change = norm(obj.curr_defense_strategy - obj.prev_defense_strategy, 2);
            nash_convergence = min(2.0, (attack_change + defense_change) / 2);
            
            if isnan(nash_convergence) || isinf(nash_convergence)
                nash_convergence = 1.0;
            end
        end
        
        function attack_coverage = calculateAttackCoverage(obj, defense_deployment, attack_success, detection_result)
            %CALCULATEATTACKCOVERAGE 计算攻击覆盖率
            
            if sum(defense_deployment) > 0
                defense_strength = defense_deployment / sum(defense_deployment);
            else
                defense_strength = zeros(size(defense_deployment));
            end
            
            detection_bonus = 0;
            if isfield(detection_result, 'detected') && detection_result.detected && ...
               isfield(detection_result, 'is_false_positive') && ~detection_result.is_false_positive
                detection_bonus = 0.3;
            end
            
            defense_effectiveness = sum(defense_strength .* obj.station_values);
            attack_failure_bonus = double(~attack_success) * 0.2;
            
            if length(defense_strength) > 1
                defense_balance = 1 - std(defense_strength);
            else
                defense_balance = 1.0;
            end
            
            attack_coverage = min(0.95, max(0.05, ...
                defense_effectiveness * 0.4 + detection_bonus + attack_failure_bonus + defense_balance * 0.1));
            
            if isnan(attack_coverage) || isinf(attack_coverage)
                attack_coverage = 0.5;
            end
        end
        
        function defense_effectiveness = calculateDefenseEffectiveness(obj, defense_deployment, damage, attack_success)
            %CALCULATEDEFENSEEFFECTIVENESS 计算防御有效性
            
            max_possible_damage = max(obj.station_values);
            if max_possible_damage > 0
                damage_prevention = max(0, (max_possible_damage - damage) / max_possible_damage);
            else
                damage_prevention = 0;
            end
            
            attack_prevention = double(~attack_success);
            
            if obj.total_resources > 0
                resource_efficiency = min(1.0, sum(defense_deployment) / obj.total_resources);
            else
                resource_efficiency = 0;
            end
            
            defense_effectiveness = 0.5 * damage_prevention + 0.3 * attack_prevention + 0.2 * resource_efficiency;
            defense_effectiveness = max(0, min(1, defense_effectiveness));
            
            if isnan(defense_effectiveness) || isinf(defense_effectiveness)
                defense_effectiveness = 0.5;
            end
        end
        
        function info = createStepInfo(obj, attack_success, damage, attacker_target, ...
                                     defender_deployment, detection_result, reward_def, reward_att)
            %CREATESTEPINFO 创建步骤信息
            
            info = struct();
            info.attack_success = attack_success;
            info.damage = damage;
            info.attacker_target = attacker_target;
            info.defender_deployment = defender_deployment;
            info.time_step = obj.time_step;
            info.radi_score = obj.radi_score;
            info.detection_result = detection_result;
            info.reward_def = reward_def;
            info.reward_att = reward_att;
            
            % 最近指标
            info.recent_success_rate = obj.computeRecentSuccessRate();
            info.recent_detection_rate = obj.computeRecentDetectionRate();
            info.recent_radi = obj.computeRecentRADI();
            
            % 增强指标
            if ~isempty(obj.nash_convergence_history)
                info.current_nash_convergence = obj.nash_convergence_history(end);
            else
                info.current_nash_convergence = 1.0;
            end
            
            if ~isempty(obj.attack_coverage_history)
                info.current_attack_coverage = obj.attack_coverage_history(end);
            else
                info.current_attack_coverage = 0.5;
            end
            
            if ~isempty(obj.defense_effectiveness_history)
                info.current_defense_effectiveness = obj.defense_effectiveness_history(end);
            else
                info.current_defense_effectiveness = 0.5;
            end
            
            % 资源分配信息
            if sum(defender_deployment) > 0
                info.resource_allocation = defender_deployment / sum(defender_deployment);
            else
                info.resource_allocation = ones(1, obj.n_stations) / obj.n_stations;
            end
        end
        
        function info = createDefaultInfo(obj)
            %CREATEDEFAULTINFO 创建默认信息
            
            info = struct();
            info.attack_success = false;
            info.damage = 0;
            info.attacker_target = 1;
            info.defender_deployment = ones(1, obj.n_stations) * (obj.total_resources / obj.n_stations);
            info.detection_result = struct('detected', false, 'detection_prob', 0, 'is_false_positive', false);
            info.reward_def = 0;
            info.reward_att = 0;
            info.radi_score = 0.5;
            info.time_step = obj.time_step;
            info.current_nash_convergence = 1.0;
            info.current_attack_coverage = 0.5;
            info.current_defense_effectiveness = 0.5;
            info.resource_allocation = ones(1, obj.n_stations) / obj.n_stations;
            info.recent_success_rate = 0;
            info.recent_detection_rate = 0;
            info.recent_radi = 0.5;
        end
    end
    
    %% ========== 静态方法（与ConfigManager配合） ==========
    methods (Static)
        function demo()
            %DEMO 演示TCSEnvironment的使用
            
            fprintf('=== TCSEnvironment v4.3 架构优化版演示 ===\n\n');
            
            % 使用ConfigManager获取配置
            if exist('ConfigManager', 'class') == 8
                config = ConfigManager.getDefaultConfig();
                fprintf('✓ 使用ConfigManager加载配置\n');
            else
                % 简单的备用配置
                config = struct();
                config.system = struct('n_stations', 3, 'n_components_per_station', [2, 2, 2], 'total_resources', 50);
                config.debug = struct('debug_mode', true);
                fprintf('⚠️  ConfigManager不可用，使用简化配置\n');
            end
            
            % 创建环境
            env = TCSEnvironment(config);
            
            fprintf('\n1. 基础功能测试...\n');
            state = env.reset();
            fprintf('   ✓ 环境重置成功，状态维度: %d\n', length(state));
            
            fprintf('\n2. 环境交互测试...\n');
            for i = 1:5
                defender_deployment = rand(1, env.n_stations) * env.total_resources;
                defender_deployment = defender_deployment / sum(defender_deployment) * env.total_resources;
                attacker_target = randi(env.n_stations);
                
                [next_state, reward_def, reward_att, info] = env.step(defender_deployment, attacker_target);
                
                if i == 1
                    fprintf('   ✓ 首次交互: RADI=%.3f, 检测=%s\n', ...
                            info.radi_score, char(string(info.detection_result.detected)));
                end
            end
            
            fprintf('\n3. 数据摘要:\n');
            summary = env.getDataSummary();
            fprintf('   ✓ 总步数: %d\n', summary.total_steps);
            fprintf('   ✓ RADI数据点: %d\n', summary.radi_data_points);
            
            fprintf('\n✅ TCSEnvironment v4.3 演示完成！\n');
            fprintf('📋 该版本与ConfigManager和AgentFactory完全兼容\n');
        end
    end
end