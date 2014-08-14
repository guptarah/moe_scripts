function target_1inK = obt_1inK(targets,M,N)

target_1inK = zeros(M,N);
for i = 1:N
	target_1inK(targets(i),i) = 1;
end

