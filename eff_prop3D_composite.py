import time 
 
#import cupy as cp 
import numpy as np 
 
def timer(func): 
    def wrapper(self, *args, **kwargs): 
        start_time = time.time() 
        result = func(self, *args, **kwargs) 
        end_time = time.time() 
        print(f"Метод {func.__name__} работал {end_time - start_time:.2f} секунд") 
        return result 
    return wrapper 
 
class eff_prop3D():
    def __init__(self, size, grid_points, CFL, stress_initial_cond, U_initial_cond, V_initial_cond, T, strain_bound_cond): 
        self.rho=1.0 #не важно у какого материала какая плотность, 
                     #тк статич задача - пока одна плотность для всех мат

        #size 
        self.Lx=size[0] 
        self.Ly=size[1] 
        self.Lz=size[2] 
 
        #grid 
        self.Nx=grid_points[0] 
        self.Ny=grid_points[1] 
        self.Nz=grid_points[2] 
        self.Nt=grid_points[3] 
        #CFL 
        self.CFL=CFL 
        #preprocessing 
        #self.__fill_preprocessing() #для проверки задачи: куб внутри шар из другого материала(или пустота)
        self.__fill_preprocessing_kern() #для проверки задачи фрагмент 10 10 2 трехцветного керна с пустотами
        #material 
        #self.__fill_material() #для проверки задачи: куб внутри шар из другого материала
        self.__fill_material_kern()#для проверки задачи фрагмент 10 10 2 трехцветного керна с пустотами
        #initial condition 
        self.set_initial_cond( stress_initial_cond, U_initial_cond, V_initial_cond, T) 
        #boundary condition 
        self.set_boundary_cond(strain_bound_cond ) 
 
    def __fill_preprocessing(self): 
        #для dt 
        E0p=1.0 
        nu0p=0.4 
        alp0p=7.7e-5 
        #hole 
        E0n=0.0 
        nu0n=0.0 
        alp0n=0.0 

        if E0p>E0n: 
            E00=E0p
            nu00=nu0p 
        else: 
            E00=E0n 
            nu00=nu0n 
 
        self.K0 = E00 / (3.0 * (1 - 2 * nu00)) 
        self.G0 = E00 / (2.0 + 2.0 * nu00) 
        # 
        dX = self.Lx / (self.Nx - 1) 
        dY = self.Ly / (self.Ny - 1) 
        dZ = self.Lz / (self.Nz) # - 1) 
 
        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx) 
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny) 
        z = np.linspace(-self.Lz/2, self.Lz/2, self.Nz) 

        self.x, self.y, self.z = np.meshgrid(x, y, z) 
 
        self.xUx, self.yUx, self.zUx = np.meshgrid( np.linspace(-(self.Lx + dX)/2, (self.Lx + dX)/2, self.Nx+1), 
                                    np.linspace(-self.Ly/2, self.Ly/2, self.Ny), 
                                    np.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij') 
 
        self.xUy, self.yUy, self.zUy = np.meshgrid( np.linspace(-self.Lx/2, self.Lx/2, self.Nx), 
                                    np.linspace(-(self.Ly + dY)/2, (self.Ly + dY)/2, self.Ny+1), 
                                    np.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij') 
 
        self.xUz, self.yUz, self.zUz = np.meshgrid(np.linspace(-self.Lx/2, self.Lx/2, self.Nx), 
                                    np.linspace(-self.Ly/2, self.Ly/2, self.Ny), 
                                    np.linspace(-(self.Lz + dZ)/2, (self.Lz + dZ)/2, self.Nz+1), indexing='ij') 
        self.dt = self.CFL * min(dX, min(dY, dZ)) / np.sqrt((self.K0 + 4*self.G0/3) / self.rho) 
        self.damp = 4 / self.dt / self.Nx 
 
        self.dX=dX 
        self.dY=dY 
        self.dZ=dZ

    def __fill_preprocessing_kern(self): 
        #для dt
        E_pir=291.2
        nu_pir=0.16 

        #тк K0 G0 пирита > K0 G0 кальцита
        E00=E_pir
        nu00=nu_pir
         
        self.K0 = E00 / (3.0 * (1 - 2 * nu00)) 
        self.G0 = E00 / (2.0 + 2.0 * nu00) 
        # 
        dX = self.Lx / (self.Nx - 1) 
        dY = self.Ly / (self.Ny - 1) 
        dZ = self.Lz / (self.Nz - 1) 
 
        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx) 
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny) 
        z = np.linspace(-self.Lz/2, self.Lz/2, self.Nz) 

        self.x, self.y, self.z = np.meshgrid(x, y, z) 
 
        self.xUx, self.yUx, self.zUx = np.meshgrid( np.linspace(-(self.Lx + dX)/2, (self.Lx + dX)/2, self.Nx+1), 
                                    np.linspace(-self.Ly/2, self.Ly/2, self.Ny), 
                                    np.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij') 
 
        self.xUy, self.yUy, self.zUy = np.meshgrid( np.linspace(-self.Lx/2, self.Lx/2, self.Nx), 
                                    np.linspace(-(self.Ly + dY)/2, (self.Ly + dY)/2, self.Ny+1), 
                                    np.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij') 
 
        self.xUz, self.yUz, self.zUz = np.meshgrid(np.linspace(-self.Lx/2, self.Lx/2, self.Nx), 
                                    np.linspace(-self.Ly/2, self.Ly/2, self.Ny), 
                                    np.linspace(-(self.Lz + dZ)/2, (self.Lz + dZ)/2, self.Nz+1), indexing='ij') 
        self.dt = self.CFL * min(dX, min(dY, dZ)) / np.sqrt((self.K0 + 4*self.G0/3) / self.rho) 
        self.damp = 4 / self.dt / self.Nx 
 
        self.dX=dX 
        self.dY=dY 
        self.dZ=dZ

    def __fill_material_kern(self):

        E = np.ones((self.Nx, self.Ny, self.Nz)) 
        nu = np.ones((self.Nx, self.Ny, self.Nz)) 
        alp = np.ones((self.Nx, self.Ny, self.Nz)) 

        fil = open("C:/Users/nices/Downloads/Telegram Desktop/bmp_voi_.raw", 'rb')#путь к бинарному файлу 
                                                                        #в кт закодирован трехцветный керн
        #в зависимости от тресхолдов мы относим воксель из файла к тому или иному материалу
        traceHold1 = 50
        traceHold2 = 95
        traceHold3 = 168

        orig_x = 1240
        orig_y = 320
        orig_z = 320

        start_x = 0
        start_y = 0
        start_z = 0

        #по итогу из файла считается фрагмент Nx Ny Nz (тк в циклах Nx-1 Ny-1 Nz-1)
        calc_x = self.Nx+1
        calc_y = self.Ny+1
        calc_z = self.Nz+1

        dist_x = (orig_x - calc_x) // 2
        dist_y = (orig_y - calc_y) // 2
        dist_z = (orig_z - calc_z) // 2

        shift = 0
        max=0

        pos = 0
        pos = fil.tell()
        fil.seek(pos + dist_z*(orig_x-1)*(orig_y-1) + shift)
        pos = fil.tell()
        fil.seek(pos + dist_y*(orig_x-1) + shift)
        pos = fil.tell()
        fil.seek(pos + dist_x + shift)

        #segment=np.zeros((calc_x - 1 - start_x, calc_y - 1 - start_y, calc_z - 1 - start_z))       
        E_cal1=55.85
        nu_cal1=0.31
        alp_cal1=25*10**(-6)

        E_cal2=80.4
        nu_cal2=0.32
        alp_cal2=25*10**(-6)

        E_pir=291.2
        nu_pir=0.16
        alp_pir=36*10**(-6)

        for k in range(start_x,  calc_x-1 ):
            for j in range(start_y, calc_y-1 ):
                for i in range(start_z, calc_z-1):
                    data = fil.read(1)
                    correctData = data[0]
                
                    if correctData <= traceHold1:
                        E[k][j][i]=0.0
                        nu[k][j][i]=0.0
                        alp[k][j][i]=0.0
                    elif (correctData > traceHold1) and (correctData <= traceHold2):
                        E[k][j][i]=E_cal1
                        nu[k][j][i]=nu_cal1
                        alp[k][j][i]=alp_cal1
                    elif (correctData > traceHold2) and (correctData <= traceHold3):
                        E[k][j][i]=E_cal2
                        nu[k][j][i]=nu_cal2
                        alp[k][j][i]=alp_cal2
                    elif correctData > traceHold3:
                        E[k][j][i]=E_pir
                        nu[k][j][i]=nu_pir
                        alp[k][j][i]=alp_pir

                pos = fil.tell()
                fil.seek(pos + orig_x - calc_x)
            pos = fil.tell()
            fil.seek(pos + (orig_y - calc_y)*(orig_x - 1))

        fil.close()
        K = E / (3.0 * (1 - 2 * nu)) 
        G = E / (2.0 + 2.0 * nu) 
 
        self.E=E 
        self.nu=nu 
        self.alp=alp 
        self.K=K 
        self.G=G 

    def __fill_material(self): 
        E0p=1.0 
        nu0p=0.4 
        alp0p=7.7e-5 
        #hole 
        E0n=0.0 
        nu0n=0.0 
        alp0n=0.0 
        #E0n = 10.0 
        #nu0n = 0.25 
        #alp0n = 1.3e-5 
 
        radios= 0.228542449538 

        E = E0p * np.ones((self.Nx, self.Ny, self.Nz)) 
        nu = nu0p * np.ones((self.Nx, self.Ny, self.Nz)) 
        alp = alp0p * np.ones((self.Nx, self.Ny, self.Nz)) 

        indices = np.sqrt(self.x**2 + self.y**2 + self.z**2) < radios 
        E[indices] = E0n 
        nu[indices] = nu0n 
        alp[indices] = alp0n 
        K = E / (3.0 * (1 - 2 * nu)) 
        G = E / (2.0 + 2.0 * nu) 
 
        self.E=E 
        self.nu=nu 
        self.alp=alp 
        self.K=K 
        self.G=G 

    def set_initial_cond(self, stress_initial_cond, U_initial_cond, V_initial_cond, T): 
        self.P0=stress_initial_cond[0] 
        self.tauxx=stress_initial_cond[1] 
        self.tauyy=stress_initial_cond[2] 
        self.tauzz=stress_initial_cond[3] 
 
        self.tauxy=stress_initial_cond[4] 
        self.tauxz=stress_initial_cond[5] 
        self.tauyz=stress_initial_cond[6] 
 
        self.Ux=U_initial_cond[0] 
        self.Uy=U_initial_cond[1] 
        self.Uz=U_initial_cond[2] 
 
        self.Vx=V_initial_cond[0] 
        self.Vy=V_initial_cond[1] 
        self.Vz=V_initial_cond[2] 
 
        self.T=T 
 
    def set_boundary_cond(self, strain_bound_cond ): 
        self.dUxdx=strain_bound_cond[0] 
        self.dUydy=strain_bound_cond[1] 
        self.dUzdz=strain_bound_cond[2] 
 
        self.dUxdy=strain_bound_cond[3] 
        self.dUxdz=strain_bound_cond[4] 
        self.dUydz=strain_bound_cond[5] 
    @timer 
    def find_Keff(self): 
        self.__solving_equation() 
        Keff = np.mean(-self.P) / (self.dUxdx + self.dUydy + self.dUzdz) 
        print(f'Keff={Keff}') 
        return Keff 
 
    @timer 
    def find_alpha(self, Keff): 
        self.__solving_equation() 
        rr=np.mean(self.P) 
        alpha = rr / Keff / self.T / 2 
        print(f'Alpha={alpha}') 
        return alpha 
 
    def __av4_xy(self,A): 
        return 0.25 * (A[:-1, :-1, :] + A[:-1, 1:, :] + A[1:, :-1, :] + A[1:, 1:, :]) 
    def __av4_xz(self,A): 
        return 0.25*(A[:-1, :, :-1] + A[:-1, :, 1:] + A[1:, :, :-1] + A[1:, :, 1:]) 
    def __av4_yz(self,A): 
        return 0.25*(A[:, :-1, :-1] + A[:, :-1, 1:] + A[:, 1:, :-1] + A[:, 1:, 1:]) 
 
    def __solving_equation(self): 
        self.Ux += self.dUxdx * self.xUx + self.dUxdy * self.yUx 
        self.Uy += self.dUydy * self.yUy 
        self.Uz += self.dUzdz * self.zUz 
 
        for it in range(self.Nt): 
            # displacement divergence 
            divU = np.diff(self.Ux, axis=0) / self.dX + np.diff(self.Uy, axis=1) / self.dY + np.diff(self.Uz, axis=2) / self.dZ 
 
            # constitutive equation - Hooke's law 
            self.P = self.P0 - self.K * divU 
 
            self.tauxx = 2.0 * self.G * (np.diff(self.Ux, axis=0) / self.dX - divU/3.0) 
            self.tauyy = 2.0 * self.G * (np.diff(self.Uy, axis=1) / self.dY - divU/3.0) 
            self.tauzz = 2.0 * self.G * (np.diff(self.Uz, axis=2) / self.dZ - divU/3.0) 
 
            self.tauxy = self.__av4_xy(self.G) * (np.diff(self.Ux[1:-1,:,:], axis=1) / self.dY + np.diff(self.Uy[:,1:-1,:], axis=0) / self.dX) 
            self.tauxz = self.__av4_xz(self.G) * (np.diff(self.Ux[1:-1,:,:], axis=2) / self.dZ + np.diff(self.Uz[:,:,1:-1], axis=0) / self.dX) 
            self.tauyz = self.__av4_yz(self.G) * (np.diff(self.Uy[:,1:-1,:], axis=2) / self.dZ + np.diff(self.Uz[:,:,1:-1], axis=1) / self.dY) 
 
            # motion equation 
            dVxdt = (np.diff(-self.P[:,1:-1,1:-1] + self.tauxx[:,1:-1,1:-1], axis=0) / self.dX + 
                    np.diff(self.tauxy[:,:,1:-1], axis=1) / self.dY + 
                    np.diff(self.tauxz[:,1:-1,:], axis=2) / self.dZ) / self.rho 
            self.Vx[1:-1,1:-1,1:-1] = self.Vx[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVxdt * self.dt 
 
            dVydt = (np.diff(self.tauxy[:,:,1:-1], axis=0) / self.dX + 
                    np.diff(-self.P[1:-1,:,1:-1] + self.tauyy[1:-1,:,1:-1], axis=1) / self.dY + 
                    np.diff(self.tauyz[1:-1,:,:], axis=2) / self.dZ) / self.rho 
            self.Vy[1:-1,1:-1,1:-1] = self.Vy[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVydt * self.dt 
 
            dVzdt = (np.diff(self.tauxz[:,1:-1,:], axis=0) / self.dX + 
                    np.diff(self.tauyz[1:-1,:,:], axis=1) / self.dY + 
                    np.diff(-self.P[1:-1,1:-1,:] + self.tauzz[1:-1,1:-1,:], axis=2) / self.dZ) / self.rho 
            self.Vz[1:-1,1:-1,1:-1] = self.Vz[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVzdt * self.dt 
 
            # displacements 
            self.Ux = self.Ux + self.Vx * self.dt 
            self.Uy = self.Uy + self.Vy * self.dt 
            self.Uz = self.Uz + self.Vz * self.dt 
 
def main(): 
    #для трехцветного керна ставим Lx=Nx Ly=Ny Lz=Nz
    Lx = 10
    Ly = 10
    Lz = 2
    size=[Lx,Ly,Lz] 
 
    # NUMERICS 
    Nx = 10
    Ny = 10
    Nz = 2

    Nt = 1000 
    grid_points=[Nx, Ny, Nz, Nt] 
    CFL = 0.25 
 
    #Material 
    #пока задаю прямо внутри класса явно чтобы проще было проверять работоспособность на разных материалах

    # INITIAL CONDITIONS 
    P0 = np.zeros((Nx, Ny, Nz)) 
    Ux = np.zeros((Nx + 1, Ny, Nz)) 
    Uy = np.zeros((Nx, Ny + 1, Nz)) 
    Uz = np.zeros((Nx, Ny, Nz + 1)) 
 
    Vx = np.zeros((Nx + 1, Ny, Nz)) 
    Vy = np.zeros((Nx, Ny + 1, Nz)) 
    Vz = np.zeros((Nx, Ny, Nz + 1)) 
 
    tauxx = np.zeros((Nx, Ny, Nz)) 
    tauyy = np.zeros((Nx, Ny, Nz)) 
    tauzz = np.zeros((Nx, Ny, Nz)) 
 
    tauxy = np.zeros((Nx - 1, Ny - 1, Nz)) 
    tauxz = np.zeros((Nx - 1, Ny, Nz - 1)) 
    tauyz = np.zeros((Nx, Ny - 1, Nz - 1)) 
 
    stress_initial_cond = [P0, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz] 
 
    U_initial_cond = [Ux, Uy, Uz] 
    V_initial_cond = [Vx, Vy, Vz] 
 
    T = 1.0 
    # BOUNDARY CONDITIONS 
    loadValue = 0.002 
    loadType=[1, 0, 0, 0, 0, 0] 
 
    dUxdx = loadValue * loadType[0] 
    dUydy = loadValue * loadType[1] 
    dUzdz = loadValue * loadType[2] 
 
    dUxdy = loadValue * loadType[3] 
    dUxdz = loadValue * loadType[4] 
    dUydz = loadValue * loadType[5] 
 
    strain_bound_cond=[dUxdx, dUydy, dUzdz, dUxdy, dUxdz, dUydz] 
 
    # 
    ob=eff_prop3D(size, grid_points, CFL, stress_initial_cond, U_initial_cond, V_initial_cond, T, strain_bound_cond) 
 
    Keff=ob.find_Keff() 
    # 
    #T=1.0 
    P0 = ob.K * ob.alp * 2 * ob.T 
    Ux = np.zeros((Nx + 1, Ny, Nz)) 
    Uy = np.zeros((Nx, Ny + 1, Nz)) 
    Uz = np.zeros((Nx, Ny, Nz + 1)) 
 
    Vx = np.zeros((Nx + 1, Ny, Nz)) 
    Vy = np.zeros((Nx, Ny + 1, Nz)) 
    Vz = np.zeros((Nx, Ny, Nz + 1)) 
 
    tauxx = np.zeros((Nx, Ny, Nz)) 
    tauyy = np.zeros((Nx, Ny, Nz)) 
    tauzz = np.zeros((Nx, Ny, Nz)) 
 
    tauxy = np.zeros((Nx - 1, Ny - 1, Nz)) 
    tauxz = np.zeros((Nx - 1, Ny, Nz - 1)) 
    tauyz = np.zeros((Nx, Ny - 1, Nz - 1)) 
 
    stress_initial_cond = [P0, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz] 
 
    U_initial_cond = [Ux, Uy, Uz] 
    V_initial_cond = [Vx, Vy, Vz] 
    # 
    ob.set_initial_cond( stress_initial_cond, U_initial_cond, V_initial_cond, T) 
    # 
    loadValue = 0.0 
    loadType=[1, 0, 0, 0, 0, 0] 
 
    dUxdx = loadValue * loadType[0] 
    dUydy = loadValue * loadType[1] 
    dUzdz = loadValue * loadType[2] 
 
    dUxdy = loadValue * loadType[3] 
    dUxdz = loadValue * loadType[4] 
    dUydz = loadValue * loadType[5] 
 
    strain_bound_cond=[dUxdx, dUydy, dUzdz, dUxdy, dUxdz, dUydz] 
    # 
    ob.set_boundary_cond(strain_bound_cond) 
 
    al=ob.find_alpha(Keff) 
 
if __name__ == "__main__": 
    main()