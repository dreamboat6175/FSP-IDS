%% Logger.m - 增强版日志记录器类
% =========================================================================
% 描述: 支持静态方法调用的单例日志记录器
% 版本: v2.0 - 修复静态方法调用问题
% =========================================================================

classdef Logger < handle
    
    properties (Access = private)
        log_file        % 日志文件路径
        fid             % 文件句柄
        log_level       % 日志级别
        is_initialized  % 初始化状态
    end
    
    properties (Access = private, Constant)
        % 日志级别定义
        LEVELS = struct('ERROR', 1, 'WARNING', 2, 'INFO', 3, 'DEBUG', 4);
    end
    
    methods (Access = private)
        function obj = Logger()
            % 私有构造函数，确保单例模式
            obj.is_initialized = false;
            obj.fid = -1;
            obj.log_level = 3; % 默认INFO级别
        end
    end
    
    methods (Static)
        function instance = getInstance()
            % 获取Logger单例实例
            persistent logger_instance;
            
            if isempty(logger_instance) || ~isvalid(logger_instance)
                logger_instance = Logger();
            end
            
            instance = logger_instance;
        end
        
        function initialize(log_file, log_level)
            % 初始化日志系统
            % 输入:
            %   log_file - 日志文件路径
            %   log_level - 日志级别 ('ERROR', 'WARNING', 'INFO', 'DEBUG')
            
            if nargin < 2
                log_level = 'INFO';
            end
            
            logger = Logger.getInstance();
            
            % 如果已经初始化，先关闭之前的文件
            if logger.is_initialized && logger.fid ~= -1
                fclose(logger.fid);
            end
            
            logger.log_file = log_file;
            logger.log_level = Logger.LEVELS.(upper(log_level));
            
            % 确保日志目录存在
            [log_dir, ~, ~] = fileparts(log_file);
            if ~isempty(log_dir) && ~exist(log_dir, 'dir')
                mkdir(log_dir);
            end
            
            % 打开日志文件
            logger.fid = fopen(log_file, 'a');
            if logger.fid == -1
                error('Logger:InitializationFailed', '无法创建日志文件: %s', log_file);
            end
            
            logger.is_initialized = true;
            
            % 记录初始化信息
            Logger.info('日志系统初始化完成');
            fprintf('✓ 日志系统初始化: %s\n', log_file);
        end
        
        function info(message)
            % 记录INFO级别日志
            Logger.writeLog('INFO', message);
        end
        
        function warning(message)
            % 记录WARNING级别日志
            Logger.writeLog('WARNING', message);
        end
        
        function error(message)
            % 记录ERROR级别日志
            Logger.writeLog('ERROR', message);
        end
        
        function debug(message)
            % 记录DEBUG级别日志
            Logger.writeLog('DEBUG', message);
        end
        
        function log(level, message)
            % 通用日志记录方法
            % 输入:
            %   level - 日志级别
            %   message - 日志消息
            Logger.writeLog(level, message);
        end
        
        function close()
            % 关闭日志系统
            logger = Logger.getInstance();
            
            if logger.is_initialized
                Logger.info('日志系统关闭');
                
                if logger.fid ~= -1
                    fclose(logger.fid);
                    logger.fid = -1;
                end
                
                logger.is_initialized = false;
                fprintf('✓ 日志系统已关闭\n');
            end
        end
        
        function flush()
            % 强制刷新日志缓冲区
            logger = Logger.getInstance();
            
            if logger.is_initialized && logger.fid ~= -1
                % MATLAB中没有直接的flush，但重新打开文件可以确保写入
                fclose(logger.fid);
                logger.fid = fopen(logger.log_file, 'a');
            end
        end
        
        function status = isInitialized()
            % 检查日志系统是否已初始化
            logger = Logger.getInstance();
            status = logger.is_initialized;
        end
        
        function setLevel(level)
            % 设置日志级别
            % 输入: level - 日志级别字符串
            
            logger = Logger.getInstance();
            
            if ischar(level) && isfield(Logger.LEVELS, upper(level))
                logger.log_level = Logger.LEVELS.(upper(level));
                Logger.info(sprintf('日志级别设置为: %s', upper(level)));
            else
                Logger.warning(sprintf('无效的日志级别: %s', level));
            end
        end
        
        function level = getLevel()
            % 获取当前日志级别
            logger = Logger.getInstance();
            
            level_names = fieldnames(Logger.LEVELS);
            for i = 1:length(level_names)
                if Logger.LEVELS.(level_names{i}) == logger.log_level
                    level = level_names{i};
                    return;
                end
            end
            level = 'UNKNOWN';
        end
    end
    
    methods (Static, Access = private)
        function writeLog(level, message)
            % 内部日志写入方法
            % 输入:
            %   level - 日志级别字符串
            %   message - 日志消息
            
            logger = Logger.getInstance();
            
            % 检查是否初始化
            if ~logger.is_initialized
                % 如果未初始化，使用默认控制台输出
                timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
                fprintf('[%s] [%s] %s\n', timestamp, level, message);
                return;
            end
            
            % 检查日志级别
            if ~isfield(Logger.LEVELS, level)
                level = 'INFO';
            end
            
            level_num = Logger.LEVELS.(level);
            if level_num > logger.log_level
                return; % 级别不够，不记录
            end
            
            % 生成时间戳
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            
            % 写入文件
            if logger.fid ~= -1
                fprintf(logger.fid, '[%s] [%s] %s\n', timestamp, level, message);
            end
            
            % 同时输出到控制台
            switch level
                case 'ERROR'
                    fprintf(2, '[%s] [ERROR] %s\n', timestamp, message);
                case 'WARNING'
                    fprintf('[%s] [WARNING] %s\n', timestamp, message);
                case 'INFO'
                    fprintf('[%s] [INFO] %s\n', timestamp, message);
                case 'DEBUG'
                    fprintf('[%s] [DEBUG] %s\n', timestamp, message);
                otherwise
                    fprintf('[%s] [%s] %s\n', timestamp, level, message);
            end
        end
    end
    
    methods (Access = protected)
        function delete(obj)
            % 析构函数，确保文件正确关闭
            if obj.fid ~= -1
                fclose(obj.fid);
            end
        end
    end
end