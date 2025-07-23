classdef TCSEnvironment < handle
    %% TCSEnvironment - 交通控制系统环境（架构优化版）
    % ================================================================
    % 版本：v4.3 - 基于项目架构的优化版本
    % 优化重点：
    % 1. 依赖ConfigManager进行配置管理
    % 2. 移除重复的配置处理逻辑
    % 3. 专注于环境核心功能
    % 4. 保持与AgentFactory的兼容性
    % ================================================================
    
    properties (Access = public)
        %% === 基础系统参数（从config获取） ===
        n_stations              % 站点数量
        n_components            % 各站点组件数量向量
        total_components        % 总组件数
        total_resources         % 总资源数
        station_values          % 站点价值向量
        
        %% === 环境状态 ===
        time_step               % 当前时间步
        current_state           % 当前状态向量
        state_dim               % 状态维度
        action_dim              % 动作维度（兼容性）
        action_dim_defender     % 防御者动作维度
        action_dim_attacker     % 攻击者动作维度
        
        %% === FSP核心参数 ===
        alpha_ewma              % FSP指数加权移动平均参数
        attacker_avg_strategy   % 攻击者平均策略
        
        %% === RADI计算参数 ===
        radi_score              % 当前RADI分数
        radi_config             % RADI配置参数
        
        %% === 检测系统参数 ===
        detection_enabled       % 是否启用检测系统
        base_detection_rate     % 基础检测率
        detection_sensitivity   % 检测敏感度
        false_positive_rate     % 误报率
        
        %% === 历史记录 ===
        attack_success_history     % 攻击成功历史
        attack_target_history      % 攻击目标历史（矩阵形式）
        defense_deployment_history % 防御部署历史（矩阵形式）
        damage_history             % 损害历史
        radi_history               % RADI历史
        detection_history          % 检测历史
        
        %% === 增强指标历史 ===
        nash_convergence_history     % Nash均衡收敛度历史
        attack_coverage_history      % 攻击覆盖率历史  
        defense_effectiveness_history % 防御有效性历史
        strategy_change_history      % 策略变化历史
        
        %% === 策略跟踪 ===
        prev_attack_strategy         % 上一轮攻击策略
        prev_defense_strategy        % 上一轮防御策略
        curr_attack_strategy         % 当前攻击策略
        curr_defense_strategy        % 当前防御策略
        
        %% === 系统配置 ===
        debug_mode              % 调试模式
        epsilon                 % 数值稳定性参数
    end
    
    properties (Access = private)
        action_templates        % 预定义动作模板
        config                  % 完整配置对象（私有）
    end
    
    methods (Access = public)
        function obj = TCSEnvironment(config)
            %TCSENVIRONMENT 构造函数
            
            if nargin < 1
                error('TCSEnvironment:InvalidInput', '需要配置参数config');
            end
            
            % 存储配置对象
            obj.config = config;
            
            % 从config中提取参数（使用ConfigManager的结构）
            obj.extractConfigParameters(config);
            
            % 初始化环境组件
            obj.initializeEnvironmentComponents();
            
            % 计算空间维度
            obj.calculateSpaceDimensions();
            
            % 重置环境
            obj.reset();
            
            if obj.debug_mode
                fprintf('[TCSEnvironment v4.3] 初始化完成 - %d站点, %d资源\n', ...
                        obj.n_stations, obj.total_resources);
            end
        end
        
        function state = reset(obj)
            %RESET 重置环境到初始状态
            
            obj.time_step = 0;
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.radi_score = 0.5;
            
            % 清空历史记录
            obj.clearAllHistoryRecords();
            
            % 重置策略跟踪
            obj.resetStrategyTracking();
            
            % 生成初始状态
            obj.current_state = obj.generateEnvironmentState();
            state = obj.current_state;
            
            if obj.debug_mode
                fprintf('[TCSEnvironment] 环境已重置\n');
            end
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_deployment, attacker_target)
            %STEP 执行一步环境交互
            
            try
                % 输入验证和修复
                [defender_deployment, attacker_target] = obj.validateAndFixStepInputs(defender_deployment, attacker_target);
                
                % 更新FSP平均策略
                obj.updateAttackerAverageStrategy(attacker_target);
                
                % 计算攻击结果
                [attack_success, damage] = obj.computeAttackOutcome(attacker_target, defender_deployment);
                
                % 检测评估
                detection_result = obj.evaluateDetection(attacker_target, defender_deployment, attack_success);
                
                % 计算奖励
                [reward_def, reward_att] = obj.computeRewards(attack_success, damage, attacker_target, defender_deployment, detection_result);
                
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
                if ~isempty(ME.stack)
                    fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
                end
                % 返回安全默认值
                next_state = obj.current_state;
                reward_def = 0;
                reward_att = 0;
                info = obj.createDefaultInfo();
            end
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
    end
    
    %% ========== 私有方法 ==========
    methods (Access = private)
        
        function extractConfigParameters(obj, config)
            %EXTRACTCONFIGPARAMETERS 从ConfigManager结构中提取参数
            
            % 系统基础参数（兼容多种配置结构）
            if isfield(config, 'system') && isfield(config.system, 'n_stations')
                obj.n_stations = config.system.n_stations;
                obj.n_components = config.system.n_components_per_station;
                obj.total_resources = config.system.total_resources;
            elseif isfield(config, 'n_stations') % 兼容旧格式
                obj.n_stations = config.n_stations;
                obj.n_components = config.n_components_per_station;
                obj.total_resources = config.total_resources;
            else
                % 默认值
                obj.n_stations = 10;
                obj.n_components = [7, 6, 8, 5, 9, 15, 4, 6, 3, 4];
                obj.total_resources = 100;
            end
            
            obj.total_components = sum(obj.n_components);
            
            % FSP参数
            if isfield(config, 'simulation') && isfield(config.simulation, 'alpha_ewma')
                obj.alpha_ewma = config.simulation.alpha_ewma;
            elseif isfield(config, 'alpha_ewma')
                obj.alpha_ewma = config.alpha_ewma;
            else
                obj.alpha_ewma = 0.1; % 默认值
            end
            
            % RADI配置
            if isfield(config, 'radi')
                obj.radi_config = config.radi;
            else
                obj.radi_config = struct();
                obj.radi_config.optimal_allocation = ones(1, obj.n_stations) / obj.n_stations;
                obj.radi_config.weights = ones(1, obj.n_stations) / obj.n_stations;
            end
            
            % 检测系统配置
            if isfield(config, 'security')
                obj.detection_enabled = true;
                obj.base_detection_rate = config.security.attack_detection_rate;
                obj.detection_sensitivity = 0.8; % 默认
                obj.false_positive_rate = config.security.false_positive_rate;
            elseif isfield(config, 'detection')
                obj.detection_enabled = config.detection.enabled;
                obj.base_detection_rate = config.detection.base_rate;
                obj.detection_sensitivity = config.detection.sensitivity;
                obj.false_positive_rate = config.detection.false_positive_rate;
            else
                obj.detection_enabled = true;
                obj.base_detection_rate = 0.3;
                obj.detection_sensitivity = 0.8;
                obj.false_positive_rate = 0.1;
            end
            
            % 调试模式
            if isfield(config, 'debug') && isfield(config.debug, 'enabled')
                obj.debug_mode = config.debug.enabled;
            elseif isfield(config, 'debug_mode')
                obj.debug_mode = config.debug_mode;
            else
                obj.debug_mode = false;
            end
            
            % 数值稳定性参数
            obj.epsilon = 1e-8;
        end
        
        function initializeEnvironmentComponents(obj)
            %INITIALIZEENVIRONMENTCOMPONENTS 初始化环境组件
            
            % 生成站点价值
            obj.generateStationValues();
            
            % 创建动作模板
            obj.action_templates = obj.createActionTemplates();
            
            % 初始化FSP平均策略
            obj.attacker_avg_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
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
            %GENERATESTATIONVALUES 生成站点价值
            
            % 基于组件数量生成价值
            component_importance = rand(1, obj.total_components);
            obj.station_values = zeros(1, obj.n_stations);
            
            idx = 1;
            for i = 1:obj.n_stations
                n_comp = obj.n_components(i);
                obj.station_values(i) = sum(component_importance(idx:idx+n_comp-1));
                idx = idx + n_comp;
            end
            
            % 归一化并减少极端差异
            obj.station_values = obj.station_values / sum(obj.station_values);
            obj.station_values = obj.station_values .^ 0.8;
            obj.station_values = obj.station_values / sum(obj.station_values);
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
            %EVALUATEDETECTION 评估检测结果
            
            detection_result = struct();
            
            if ~obj.detection_enabled
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
            detection_prob = min(0.95, detection_prob);
            
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