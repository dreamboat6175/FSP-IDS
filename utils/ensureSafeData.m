function safe_data = ensureSafeData(data, expected_size, default_value)
%ENSURESAFEDATA 确保数据维度安全
if nargin < 3
    default_value = 0;
end

if isempty(data)
    safe_data = repmat(default_value, expected_size);
    return;
end

try
    if isvector(data) && length(data) == prod(expected_size)
        safe_data = reshape(data, expected_size);
    elseif all(size(data) == expected_size)
        safe_data = data;
    else
        safe_data = repmat(default_value, expected_size);
        if numel(data) > 0
            copy_size = min(numel(safe_data), numel(data));
            safe_data(1:copy_size) = data(1:copy_size);
        end
    end
catch
    safe_data = repmat(default_value, expected_size);
end
end