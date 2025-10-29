function index = f_findPosition(position, lon, lat)
% 查询点的index

index = zeros(size(position, 1), 1);
for i = 1: size(position, 1)
    DistanceSquare = (position(i,1)-lon).^2 + (position(i,2)-lat).^2;
    [~,index(i)] = min(DistanceSquare);
end



end

