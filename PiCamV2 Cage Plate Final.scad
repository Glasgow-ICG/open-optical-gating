////////////////////////////////////////////////////////////////////////
// Lens Holder for Thor Lab Equipment
////////////////////////////////////////////////////////////////////////

/// PARAMETERS FOR THE CUSTOMIZER ///

L_r = 15.5; // Lens Width
S_r = 2; // Holding Screw Width (Defined here for M4 Bolts)

//Lens Support Dimension
L_x = 41;
L_y = 41;
L_z = 5;

pcb_x = 25.2;
pcb_y = 24;
pcb_z = 3;

////////////////////////////////////////////////////////////////////////

Rods = 15; // Rods Relative x/y Position to Center as defined by Thor Lab
R_r = 3.6; // Rods Radius as defined by Thor Lab
s_r = 1.25; // Rod Holding Screw Radius as defined by Thor Lab

s2_r = 1.1; //Screw radius to attach picam
x_sep = 21;
y_zep = 12.5;

rim = 2.2;
rim2 = 9.5;

cab_w = 20;
cab_z = 5;
cab_y = 6.1;

cut_y = cab_y+(L_y-pcb_y)/2;
cut_z = cab_z+pcb_z; 




module Base_pos(){
   cube ([L_x,L_y,L_z], center=true);   
}
module Base_neg(){
    
    //Centre cutout
    translate([0,0,L_z/2-pcb_z/2]){cube ($fn=50, [pcb_x,pcb_y,pcb_z], center = true);}
    
    //Camera screw holes
    translate([(pcb_x/2-rim),pcb_y/2-rim,-L_z/2]){cylinder ($fn=100, r=s2_r, h=L_z);}
    
    translate([-(pcb_x/2-rim),pcb_y/2-rim,-L_z/2]){cylinder ($fn=100, r=s2_r, h=L_z);}
    
    translate([(pcb_x/2-rim),-(pcb_y/2-rim2),-L_z/2]){cylinder ($fn=100, r=s2_r, h=L_z);}
    
    translate([-(pcb_x/2-rim),-(pcb_y/2-rim2),-L_z/2]){cylinder ($fn=100, r=s2_r, h=L_z);}
    
    
    //Rod holes
    translate([-Rods,-Rods,-L_z/2]){cylinder ($fn=100, r=R_r, h=L_z);}
     translate([+Rods,-Rods,-L_z/2]){cylinder ($fn=100, r=R_r, h=L_z);}
      translate([-Rods,Rods,-L_z/2]){cylinder ($fn=100, r=R_r, h=L_z);}
       translate([+Rods,Rods,-L_z/2]){cylinder ($fn=100, r=R_r, h=L_z);}
    
       
    /*Rod screw holes
      translate([Rods,L_y/2,0]){rotate ([90,0,0])cylinder ($fn=100, r=s_r, h=L_y/2-Rods);}
       translate([-Rods,L_y/2,0]){rotate ([90,0,0])cylinder ($fn=100, r=s_r, h=L_y/2-Rods);} 
        translate([Rods,-L_y/2,0]){rotate ([270,0,0])cylinder ($fn=100, r=s_r, h=L_y/2-Rods);}
         translate([-Rods,-L_y/2,0]){rotate ([270,0,0])cylinder ($fn=100, r=s_r, h=L_y/2-Rods);}
    */
       
    //Cable Cutout
         translate([0,(cut_y-L_y)/2,(cut_z+L_z)/2-cut_z]) {cube ($fn = 50, [cab_w,cut_y,cut_z],center = true);}

}
difference() {
    Base_pos();
    Base_neg();
}

    
   