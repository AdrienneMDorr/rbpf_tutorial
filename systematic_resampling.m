function indices = systematic_resampling(wk)
nP = numel(wk);%number of particles.
v = rand();%random number to generate the random positions on the cells.
indices = zeros(size(wk));
    %Generate the points that will be along the grid of cummulative sum diagram.
    u = zeros(1,nP);
    for j = 1:nP
        %Distributes nP-1 points along the cumulative sum.
        %Note that we add the random offset v to each point.
        u(j) = ((j-1)+v)/nP;
      %For example, with no random v the distribution could look like the
    %following
 %u(1)= 0  u(2)= 1/nP | u(2)= 2/nP | u(3)= 3/nP  u(4)= 4/nP | ...  u(nP-1)= nP-1/nP
   %Now let's see which weights fall into each division
    end
    
    z = cumsum(wk); %Compute cummulative sum of the weights.
    %The cumulative sum will generate numbers that can be used for data
    %segmentation
    
   i =1;
   j = 1;
while j <= nP
    %We keep assigning particles to the ith division, until there is one
    %weight that is outside the division.
   if u(j)< z(i) % z(i) is the ith division in the cumulative sum.
       %This line copies the particle into a new array. It will keep
       %copying the same particle to the new array until no more weights
       %appear in the current ith division.
           indices(j) = i;
           j = j + 1;
   else
       %We enter here when we need to move to the next division.
       i = i + 1;
   end
end