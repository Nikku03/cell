function r = mcfit3(par0,xx,yy,zz)

%Fit a plane to the data given by the "predict" command 
  
  A1=par0(1);
  A2=par0(2);
  A3=par0(3);
  A4=par0(4);
    
  mc = A1*xx + A2*yy + A3*zz + A4;
  r = sum( mc.^2 );

   end %function
