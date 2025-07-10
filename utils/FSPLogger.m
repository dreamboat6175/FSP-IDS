% FSPLogger.m
classdef FSPLogger < handle
    properties
        log_file
        fid
        log_level
    end
    
    methods
        function obj = FSPLogger(log_file, log_level)
            if nargin < 2
                log_level = 'INFO';
            end
            obj.log_file = log_file;
            obj.log_level = log_level;
            
            % 确保日志目录存在
            [log_dir, ~, ~] = fileparts(log_file);
            if ~exist(log_dir, 'dir')
                mkdir(log_dir);
            end
            
            obj.fid = fopen(log_file, 'a');
            if obj.fid == -1
                error('无法创建日志文件: %s', log_file);
            end
            obj.info('日志系统初始化');
        end
        
        function log(obj, level, message)
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            fprintf(obj.fid, '[%s] [%s] %s\n', timestamp, level, message);
            fprintf('[%s] %s\n', level, message);
        end
        
        function info(obj, message)
            obj.log('INFO', message);
        end
        
        function warning(obj, message)
            obj.log('WARNING', message);
        end
        
        function error(obj, message)
            obj.log('ERROR', message);
        end
        
        function delete(obj)
            if obj.fid ~= -1
                fclose(obj.fid);
            end
        end
    end
end