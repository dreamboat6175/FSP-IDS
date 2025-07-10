%% ConfigManager.m - 配置管理器类
% =========================================================================
% 描述: 负责配置文件的加载、验证和管理
% =========================================================================

classdef ConfigManager
    
    methods (Static)
        function config = loadConfig(filename)
            % 加载配置文件
            % 输入: filename - 配置文件名（JSON格式）
            % 输出: config - 配置结构体
            
            if nargin < 1 || isempty(filename)
                filename = 'default_config.json';
            end
            
            config_path = fullfile('config', filename);
            
            % 检查配置文件是否存在
            if exist(config_path, 'file')
                try
                    % 读取JSON文件
                    config_text = fileread(config_path);
                    config = jsondecode(config_text);
                    fprintf('✓ 配置文件加载成功: %s\n', filename);
                catch ME
                    warning('配置文件读取失败: %s', ME.message);
                    config = ConfigManager.getDefaultConfig();
                end
            else
                fprintf('配置文件不存在，使用默认配置\n');
                config = ConfigManager.getDefaultConfig();
                % 保存默认配置
                ConfigManager.saveConfig(config, 'default_config.json');
            end
            
            % 确保所有必要字段存在
            config = ConfigManager.ensureRequiredFields(config);
        end
        
        function config = getDefaultConfig()
            % 获取默认配置
            
            % 系统配置
            config.n_stations = 5;
            config.n_components_per_station = [7, 6, 8, 5, 9];  % 每个主站的组件数
            
            % ===== 修改开始: 采纳报告建议优化参数 =====
            % FSP参数
            config.n_iterations = 1500; % 增加迭代次数以确保收敛
            config.n_episodes_per_iter = 100;
            config.pool_size_limit = 50;
            config.pool_update_interval = 10;
            
            % 学习参数
            config.learning_rate = 0.15;      % 提高学习率至建议范围
            config.discount_factor = 0.95;
            config.epsilon = 0.4;             % 提高初始探索率至建议范围
            config.epsilon_decay = 0.999;     % 更缓慢的衰减
            config.epsilon_min = 0.05;        % 提高最小探索率
            config.temperature = 1.0;
            config.temperature_decay = 0.995;
            % ===== 修改结束 =====
            
            % 算法选择
            config.algorithms = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
            
            % 攻击配置
            config.attack_types = {'wireless_spoofing', 'dos_attack', 'semantic_attack', ...
                                  'supply_chain', 'protocol_exploit', 'maintenance_port'};
            config.attack_severity = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8];
            config.attack_detection_difficulty = [0.4, 0.3, 0.7, 0.8, 0.6, 0.5];
            
            % 资源配置
            config.resource_types = {'computation', 'bandwidth', 'sensors', ...
                                   'scanning_freq', 'inspection_depth'};
            config.resource_effectiveness = [0.7, 0.6, 0.8, 0.5, 0.9];
            config.total_resources = 100;
            
            % 输出配置
            config.display_interval = 50;
            config.save_interval = 100;
            config.param_update_interval = 50;
            config.visualization = true;
            config.generate_report = true;
            
            % 文件配置
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            config.log_file = sprintf('logs/simulation_%s.log', timestamp);
            config.results_dir = 'results';
            config.report_dir = 'reports';
            
            % 并行计算配置
            config.use_parallel = false;  % 暂时禁用并行计算
            config.num_workers = 4;
        end
        
        function config = ensureRequiredFields(config)
            % 确保所有必要字段存在
            default = ConfigManager.getDefaultConfig();
            fields = fieldnames(default);
            
            for i = 1:length(fields)
                if ~isfield(config, fields{i})
                    config.(fields{i}) = default.(fields{i});
                end
            end
        end
        
        function validateConfig(config)
            % 验证配置参数的合法性
            
            % 基本参数检查
            assert(config.n_stations > 0, '主站数量必须大于0');
            assert(length(config.n_components_per_station) == config.n_stations, ...
                   '组件数量数组长度必须等于主站数量');
            
            % 学习参数检查
            assert(config.learning_rate > 0 && config.learning_rate <= 1, ...
                   '学习率必须在(0,1]范围内');
            assert(config.discount_factor >= 0 && config.discount_factor <= 1, ...
                   '折扣因子必须在[0,1]范围内');
            assert(config.epsilon >= 0 && config.epsilon <= 1, ...
                   '探索率必须在[0,1]范围内');
            
            % 数组长度一致性检查
            n_attack_types = length(config.attack_types);
            assert(length(config.attack_severity) == n_attack_types, ...
                   '攻击严重程度数组长度与攻击类型不匹配');
            assert(length(config.attack_detection_difficulty) == n_attack_types, ...
                   '检测难度数组长度与攻击类型不匹配');
            
            n_resource_types = length(config.resource_types);
            assert(length(config.resource_effectiveness) == n_resource_types, ...
                   '资源效率数组长度与资源类型不匹配');
            
            % 创建必要的目录
            ConfigManager.createDirectories(config);
            
            fprintf('✓ 配置验证通过\n');
        end
        
        function saveConfig(config, filename)
            % 保存配置到JSON文件
            if nargin < 2
                filename = sprintf('config_backup_%s.json', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            config_path = fullfile('config', filename);
            
            try
                % 确保目录存在
                if ~exist('config', 'dir')
                    mkdir('config');
                end
                
                % 保存为JSON
                config_json = jsonencode(config, 'PrettyPrint', true);
                fid = fopen(config_path, 'w');
                fprintf(fid, '%s', config_json);
                fclose(fid);
                
                fprintf('✓ 配置已保存: %s\n', config_path);
            catch ME
                warning('配置保存失败: %s', ME.message);
            end
        end
        
        function createDirectories(config)
            % 创建必要的目录结构
            dirs = {'logs', 'results', 'reports', 'config', 'data'};
            
            for i = 1:length(dirs)
                if ~exist(dirs{i}, 'dir')
                    mkdir(dirs{i});
                end
            end
            
            % 创建特定的结果目录
            if isfield(config, 'results_dir') && ~exist(config.results_dir, 'dir')
                mkdir(config.results_dir);
            end
            
            if isfield(config, 'report_dir') && ~exist(config.report_dir, 'dir')
                mkdir(config.report_dir);
            end
        end
        
        function config = updateLearningParameters(config, iteration)
            % 自适应更新学习参数
            
            % 更新探索率
            config.epsilon = max(config.epsilon_min, ...
                               config.epsilon * config.epsilon_decay);
            
            % 更新学习率（可选）
            decay_factor = 1 / (1 + 0.001 * iteration);
            config.learning_rate = config.learning_rate * decay_factor;
            
            % 更新温度参数
            if isfield(config, 'temperature')
                config.temperature = max(0.1, config.temperature * config.temperature_decay);
            end
        end
    end
end