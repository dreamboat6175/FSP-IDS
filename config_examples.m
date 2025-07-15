%% config_examples.m - 配置示例和测试脚本
% =========================================================================
% 描述: 提供各种配置示例和测试脚本，便于快速上手和调试
% =========================================================================

%% 示例1: 基础配置
function config = getBasicConfig()
    % 基础仿真配置 - 解决维度不匹配问题
    
    config = struct();
    
    % 环境参数 - 确保所有向量长度与n_stations一致
    config.n_stations = 10;                % 主站数量
    config.n_components_per_station = 5;   % 每站点组件数
    config.total_resources = 100;          % 总防御资源
    config.random_seed = 42;               % 随机种子
    
    % 仿真参数
    config.n_episodes = 300;               % 仿真轮次
    config.max_steps_per_episode = 80;     % 每轮最大步数
    config.convergence_threshold = 0.01;   % 收敛阈值
    config.performance_check_interval = 25; % 性能检查间隔
    
    % 智能体配置
    config.agents = struct();
    
    % 攻击者智能体 - Q-Learning
    config.agents.attacker = struct();
    config.agents.attacker.type = 'QLearning';
    config.agents.attacker.learning_rate = 0.1;
    config.agents.attacker.discount_factor = 0.95;
    config.agents.attacker.epsilon = 0.3;
    config.agents.attacker.epsilon_decay = 0.995;
    config.agents.attacker.epsilon_min = 0.01;
    config.agents.attacker.learning_rate_min = 0.01;
    config.agents.attacker.learning_rate_decay = 0.9995;
    
    % 防御者智能体 - FSP
    config.agents.defender = struct();
    config.agents.defender.type = 'FSP';
    config.agents.defender.learning_rate = 0.05;
    config.agents.defender.discount_factor = 0.95;
    config.agents.defender.fsp_alpha = 0.1;
    
    % 资源和攻击类型 - 确保维度匹配
    config.resource_types = {'cpu', 'memory', 'network', 'storage', 'security'};
    config.attack_types = {'dos', 'intrusion', 'malware', 'social', 'physical', 'cyber'};
    config.n_resource_types = length(config.resource_types);
    config.n_attack_types = length(config.attack_types);
    
    % 站点价值 - 确保长度为n_stations
    config.station_values = generateStationValues(config.n_stations);
    
    % 奖励权重
    config.reward_weights = struct();
    config.reward_weights.radi = 0.5;
    config.reward_weights.damage = 0.5;
    config.reward_weights.efficiency = 0.3;
    
    % 输出配置
    config.output = struct();
    config.output.save_models = true;
    config.output.generate_plots = true;
    config.output.save_data = true;
    config.output.results_dir = fullfile(pwd, 'results');
    
    % 确保输出目录存在
    if ~exist(config.output.results_dir, 'dir')
        mkdir(config.output.results_dir);
    end

    % 在所有设置config.n_stations和config.n_components_per_station后，自动修正长度
    if isfield(config, 'n_stations') && isfield(config, 'n_components_per_station')
        n = config.n_stations;
        c = config.n_components_per_station;
        if length(c) < n
            config.n_components_per_station = [c, ones(1, n-length(c))*3];
        elseif length(c) > n
            config.n_components_per_station = c(1:n);
        end
    end
end

%% 示例2: 高性能配置
function config = getHighPerformanceConfig()
    % 高性能仿真配置
    
    config = getBasicConfig(); % 基于基础配置
    
    % 增加仿真规模
    config.n_stations = 15;
    config.n_episodes = 1000;
    config.max_steps_per_episode = 150;
    config.total_resources = 200;
    
    % 重新生成站点价值以匹配新的站点数量
    config.station_values = generateStationValues(config.n_stations);
    
    % 优化智能体参数
    config.agents.attacker.learning_rate = 0.05;
    config.agents.attacker.epsilon = 0.2;
    config.agents.defender.learning_rate = 0.03;
    config.agents.defender.fsp_alpha = 0.05;
    
    % 性能优化选项
    config.optimization = struct();
    config.optimization.use_sparse_matrices = true;
    config.optimization.memory_limit_mb = 1000;
    config.optimization.parallel_episodes = false; % 未来功能

    % 在所有设置config.n_stations和config.n_components_per_station后，自动修正长度
    if isfield(config, 'n_stations') && isfield(config, 'n_components_per_station')
        n = config.n_stations;
        c = config.n_components_per_station;
        if length(c) < n
            config.n_components_per_station = [c, ones(1, n-length(c))*3];
        elseif length(c) > n
            config.n_components_per_station = c(1:n);
        end
    end
end

%% 示例3: 调试配置
function config = getDebugConfig()
    % 调试用的小规模配置
    
    config = struct();
    
    % 小规模参数便于调试
    config.n_stations = 5;
    config.n_components_per_station = 3;
    config.total_resources = 50;
    config.random_seed = 123;
    
    % 短时间仿真
    config.n_episodes = 50;
    config.max_steps_per_episode = 20;
    config.convergence_threshold = 0.05;
    config.performance_check_interval = 10;
    
    % 简化的智能体配置
    config.agents = struct();
    config.agents.attacker = struct();
    config.agents.attacker.type = 'QLearning';
    config.agents.attacker.learning_rate = 0.2;
    config.agents.attacker.discount_factor = 0.9;
    config.agents.attacker.epsilon = 0.5;
    config.agents.attacker.epsilon_decay = 0.99;
    config.agents.attacker.epsilon_min = 0.05;
    
    config.agents.defender = struct();
    config.agents.defender.type = 'FSP';
    config.agents.defender.learning_rate = 0.1;
    config.agents.defender.discount_factor = 0.9;
    config.agents.defender.fsp_alpha = 0.2;
    
    % 资源类型
    config.resource_types = {'cpu', 'memory', 'network'};
    config.attack_types = {'dos', 'intrusion', 'malware'};
    config.n_resource_types = length(config.resource_types);
    config.n_attack_types = length(config.attack_types);
    
    % 确保站点价值维度正确
    config.station_values = generateStationValues(config.n_stations);
    
    % 简化输出
    config.output = struct();
    config.output.save_models = true;
    config.output.generate_plots = true;
    config.output.save_data = false;
    config.output.results_dir = fullfile(pwd, 'debug_results');
    
    if ~exist(config.output.results_dir, 'dir')
        mkdir(config.output.results_dir);
    end

    % 在所有设置config.n_stations和config.n_components_per_station后，自动修正长度
    if isfield(config, 'n_stations') && isfield(config, 'n_components_per_station')
        n = config.n_stations;
        c = config.n_components_per_station;
        if length(c) < n
            config.n_components_per_station = [c, ones(1, n-length(c))*3];
        elseif length(c) > n
            config.n_components_per_station = c(1:n);
        end
    end
end

%% 辅助函数：生成站点价值
function station_values = generateStationValues(n_stations)
    % 生成归一化的站点价值向量，确保长度为n_stations
    
    % 生成随机基础价值
    base_values = 0.5 + 0.5 * rand(1, n_stations);
    
    % 添加一些站点为关键站点（更高价值）
    critical_stations = randperm(n_stations, max(1, round(n_stations * 0.3)));
    base_values(critical_stations) = base_values(critical_stations) * 2;
    
    % 归一化
    station_values = base_values / sum(base_values);
    
    % 验证输出
    assert(length(station_values) == n_stations, ...
        '站点价值向量长度(%d)与站点数量(%d)不匹配', length(station_values), n_stations);
    assert(abs(sum(station_values) - 1) < 1e-10, '站点价值未正确归一化');
end

%% 测试函数：验证配置
function runConfigurationTests()
    % 运行配置验证测试
    
    fprintf('开始配置验证测试...\n');
    
    % 测试基础配置
    fprintf('测试基础配置...');
    config1 = getBasicConfig();
    validateConfiguration(config1);
    fprintf(' 通过\n');
    
    % 测试高性能配置
    fprintf('测试高性能配置...');
    config2 = getHighPerformanceConfig();
    validateConfiguration(config2);
    fprintf(' 通过\n');
    
    % 测试调试配置
    fprintf('测试调试配置...');
    config3 = getDebugConfig();
    validateConfiguration(config3);
    fprintf(' 通过\n');
    
    % 测试维度匹配
    fprintf('测试维度匹配...');
    testDimensionConsistency(config1);
    testDimensionConsistency(config2);
    testDimensionConsistency(config3);
    fprintf(' 通过\n');
    
    fprintf('所有配置验证测试通过！\n');
end

function validateConfiguration(config)
    % 验证配置参数的有效性
    
    % 基础参数检查
    assert(config.n_stations > 0, 'n_stations 必须大于 0');
    assert(all(config.n_components_per_station > 0), 'n_components_per_station 必须大于 0');
    assert(config.total_resources > 0, 'total_resources 必须大于 0');
    assert(config.n_episodes > 0, 'n_episodes 必须大于 0');
    assert(config.max_steps_per_episode > 0, 'max_steps_per_episode 必须大于 0');
    
    % 智能体参数检查
    assert(isfield(config, 'agents'), '配置缺少智能体设置');
    assert(isfield(config.agents, 'attacker'), '配置缺少攻击者设置');
    assert(isfield(config.agents, 'defender'), '配置缺少防御者设置');
    
    % 学习率检查
    assert(config.agents.attacker.learning_rate > 0 && config.agents.attacker.learning_rate <= 1, ...
           '攻击者学习率必须在 (0,1] 范围内');
    assert(config.agents.defender.learning_rate > 0 && config.agents.defender.learning_rate <= 1, ...
           '防御者学习率必须在 (0,1] 范围内');
    
    % epsilon参数检查
    assert(config.agents.attacker.epsilon >= 0 && config.agents.attacker.epsilon <= 1, ...
           '攻击者epsilon必须在 [0,1] 范围内');
    
    % 维度一致性检查
    if isfield(config, 'station_values')
        assert(length(config.station_values) == config.n_stations, ...
            '站点价值向量长度与站点数量不匹配');
    end
end

function testDimensionConsistency(config)
    % 测试维度一致性
    
    % 检查站点价值向量
    if isfield(config, 'station_values')
        assert(length(config.station_values) == config.n_stations, ...
            'station_values长度(%d)与n_stations(%d)不匹配', ...
            length(config.station_values), config.n_stations);
        
        % 检查归一化
        assert(abs(sum(config.station_values) - 1) < 1e-10, ...
            'station_values未正确归一化，和为%.6f', sum(config.station_values));
    end
    
    % 检查资源和攻击类型一致性
    assert(config.n_resource_types == length(config.resource_types), ...
        'n_resource_types与resource_types长度不匹配');
    assert(config.n_attack_types == length(config.attack_types), ...
        'n_attack_types与attack_types长度不匹配');
end

%% 快速测试函数
function runQuickTest()
    % 运行快速测试以验证系统工作正常
    
    fprintf('运行快速测试...\n');
    
    try
        % 使用调试配置运行短时间仿真
        config = getDebugConfig();
        
        % 创建环境进行基础测试
        env = TCSEnvironment(config);
        
        % 测试环境重置
        state = env.reset();
        fprintf('环境重置成功，状态维度: %d\n', length(state));
        
        % 测试环境步骤
        attacker_action = 1;
        defender_action = ones(1, config.n_stations) * config.total_resources / config.n_stations;
        
        [next_state, reward_def, reward_att, info] = env.step(attacker_action, defender_action);
        fprintf('环境步骤成功，奖励: 攻击者=%.2f, 防御者=%.2f\n', ...
            reward_att, reward_def);
        
        % 测试智能体创建
        state_dim = env.state_dim;
        action_dim = env.action_dim;
        
        attacker = QLearningAgent('test_attacker', 'QLearning', ...
            config.agents.attacker, state_dim, action_dim);
        fprintf('攻击者智能体创建成功\n');
        
        defender = FSPAgent('test_defender', 'FSP', ...
            config.agents.defender, state_dim, action_dim);
        fprintf('防御者智能体创建成功\n');
        
        % 测试动作选择
        att_action = attacker.selectAction(state);
        def_action = defender.selectAction(state);
        fprintf('动作选择成功: 攻击者=%d, 防御者=%d\n', att_action, def_action);
        
        fprintf('快速测试完成，系统工作正常！\n');
        
    catch ME
        fprintf('快速测试失败: %s\n', ME.message);
        fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
        rethrow(ME);
    end
end

%% 修复维度问题的专用函数
function fixedConfig = fixDimensionIssues(originalConfig)
    % 修复配置中的维度问题
    
    fixedConfig = originalConfig;
    
    % 确保站点价值向量长度正确
    if isfield(fixedConfig, 'station_values')
        if length(fixedConfig.station_values) ~= fixedConfig.n_stations
            fprintf('警告: 修复station_values维度从%d到%d\n', ...
                length(fixedConfig.station_values), fixedConfig.n_stations);
            fixedConfig.station_values = generateStationValues(fixedConfig.n_stations);
        end
    else
        fixedConfig.station_values = generateStationValues(fixedConfig.n_stations);
    end
    
    % 确保所有基于站点数量的向量都有正确的维度
    fields_to_check = {'n_components_per_station'};
    for i = 1:length(fields_to_check)
        field = fields_to_check{i};
        if isfield(fixedConfig, field)
            if isscalar(fixedConfig.(field))
                % 如果是标量，保持不变（每个站点相同）
                continue;
            elseif length(fixedConfig.(field)) ~= fixedConfig.n_stations
                % 如果是向量但长度不对，调整长度
                fprintf('警告: 修复%s维度\n', field);
                if length(fixedConfig.(field)) < fixedConfig.n_stations
                    % 扩展向量
                    fixedConfig.(field) = [fixedConfig.(field), ...
                        repmat(fixedConfig.(field)(end), 1, ...
                        fixedConfig.n_stations - length(fixedConfig.(field)))];
                else
                    % 截断向量
                    fixedConfig.(field) = fixedConfig.(field)(1:fixedConfig.n_stations);
                end
            end
        end
    end
    
    % 验证修复后的配置
    validateConfiguration(fixedConfig);
end

%% 使用示例
function demonstrateUsage()
    % 演示如何使用不同的配置
    
    fprintf('=== FSP仿真系统配置示例 ===\n\n');
    
    % 示例1: 基础使用
    fprintf('1. 基础配置使用:\n');
    config = getBasicConfig();
    fprintf('   站点数量: %d\n', config.n_stations);
    fprintf('   站点价值: [');
    fprintf('%.3f ', config.station_values);
    fprintf(']\n');
    fprintf('   仿真轮次: %d\n\n', config.n_episodes);
    
    % 示例2: 高性能配置
    fprintf('2. 高性能配置使用:\n');
    config_hp = getHighPerformanceConfig();
    fprintf('   站点数量: %d\n', config_hp.n_stations);
    fprintf('   仿真轮次: %d\n', config_hp.n_episodes);
    fprintf('   总资源: %d\n\n', config_hp.total_resources);
    
    % 示例3: 调试配置
    fprintf('3. 调试配置使用:\n');
    config_debug = getDebugConfig();
    fprintf('   站点数量: %d\n', config_debug.n_stations);
    fprintf('   仿真轮次: %d\n', config_debug.n_episodes);
    fprintf('   每轮步数: %d\n\n', config_debug.max_steps_per_episode);
    
    % 运行快速测试
    fprintf('4. 运行快速测试:\n');
    runQuickTest();
end

%% 主入口函数
function main()
    % 主函数 - 演示所有功能
    
    fprintf('=== 配置示例和测试脚本 ===\n\n');
    
    % 运行配置测试
    runConfigurationTests();
    fprintf('\n');
    
    % 演示使用方法
    demonstrateUsage();
    
    fprintf('\n=== 全部完成 ===\n');
end

% 如果直接运行此文件，执行主函数
if ~isdeployed
    main();
end