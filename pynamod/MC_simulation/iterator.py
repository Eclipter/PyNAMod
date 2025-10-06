import torch
import signal
import numpy as np

from pynamod.geometry.trajectories import Integrator_Trajectory, H5_Trajectory, Tensor_Trajectory

from pynamod.geometry.bp_step_geometry import Geometry_Functions
from pynamod.MC_simulation.stats_display import _Stats_Display

class Iterator:
    def __init__(self,cg_structure,energy,sigma_transl=1,sigma_rot=1):
        self.cg_structure = cg_structure
        self.energy = energy
        self.sigma_transl = sigma_transl
        self.sigma_rot = sigma_rot
        self.res_trajectory = cg_structure.dna.geom_params.trajectory
        
    
    
    def run(self,target_accepted_steps=int(1e5),max_steps=int(1e6),transfer_to_memory_every=None,save_every=1,
            mute=False,KT_factor=1,integration_mod='minimize',device='cpu',traj_init_step=None):
            
        self._prepare_system(target_accepted_steps,transfer_to_memory_every,device,integration_mod,traj_init_step,KT_factor)
        
        self._stop_loop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self._stats_display = _Stats_Display(max_steps,mute)

        while self.total_step < max_steps and self.accepted_steps < target_accepted_steps:

            self._integration_step(save_every,integration_mod)
            
            self._stats_display.show_step_data(self.accepted_steps,self.total_step,self.prev_e.sum().item())
            #Stop only when a step is completed in case of keyboard interrupt.
            if self._stop_loop:
                break
                    
        self._transfer_to_memory(steps=self.trajectory.cur_step)
        
        self._stats_display.show_final_data(target_accepted_steps,self._stop_loop,self.accepted_steps,self.total_step)
    
    def to(self,device):
        self.trajectory.to(device)
        self.energy.to(device)
    
            
    def _prepare_system(self,target_accepted_steps,transfer_to_memory_every,device,integration_mod,traj_init_step,KT_factor):
        if 'energies' not in self.res_trajectory.attrs_names:
            self.res_trajectory.add_attr('energies',(4,))
        
        if not transfer_to_memory_every:
            transfer_to_memory_every = target_accepted_steps
        self.transfer_to_memory_every = transfer_to_memory_every    
        self.total_step = self.accepted_steps = self.last_accepted = 0
        if not traj_init_step:
            traj_init_step = len(self.res_trajectory) - 1
            
        cur_step = self.res_trajectory.cur_step
        self.res_trajectory.cur_step = traj_init_step
        self._create_tens_trajectory()
        self.to(device)
        self.prev_e = torch.hstack(self.energy.get_energy_components(self.trajectory))
        self.res_trajectory._set_frame_attr('energies',self.prev_e.cpu())
        self.energy_comp_traj = torch.zeros(self.transfer_to_memory_every,4,device=device)
        self.res_trajectory.cur_step = cur_step
        
        self._set_change_indices(integration_mod)
        
        self.geom_func = Geometry_Functions()
        self.change = torch.zeros(6,dtype=self.trajectory.origins.dtype,device=device)
        self.normal_mean = torch.zeros(6,device=device)
        self.normal_scale = torch.tensor([*[self.sigma_transl]*3,*[self.sigma_rot]*3],device=device)
        self._scaled_KT = KT_factor*self.energy.KT
        
    
    
    def _set_change_indices(self,integration_mod):
        self.movable_ind = torch.arange(self.trajectory.data_len,dtype=int)[self.cg_structure.dna.movable_steps]
        if integration_mod == 'minimize':
            self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            self.movable_ind_len = self.movable_ind.shape[0]
    
    
    def _create_tens_trajectory(self):
        if isinstance(self.res_trajectory._get_frame_attr('local_params'),np.ndarray):
            init_local_params = torch.from_numpy(self.res_trajectory._get_frame_attr('local_params'))
            init_ref_frames = torch.from_numpy(self.res_trajectory._get_frame_attr('ref_frames'))
            init_ori = torch.from_numpy(self.res_trajectory._get_frame_attr('origins'))    
        else:
            init_local_params = self.cg_structure.dna.geom_params.local_params
            init_ref_frames = self.cg_structure.dna.geom_params.ref_frames
            init_ori = self.cg_structure.dna.geom_params.origins
            
        if self.cg_structure.proteins:
            init_prot_ori = torch.vstack([protein.origins for protein in self.cg_structure.proteins[::-1]])
        else:
            init_prot_ori = torch.empty((0,3))
                
        init_total_ori = torch.vstack([init_ori,init_prot_ori])
        ln = init_ref_frames.shape[0]
        traj_len = self.transfer_to_memory_every+1
        dtype = init_ref_frames.dtype
        prot_origins_ln = init_prot_ori.shape[0]

        self.trajectory = Integrator_Trajectory(self.cg_structure.proteins,dtype,traj_len,ln)
        self.trajectory.origins,self.trajectory.ref_frames = init_total_ori.to(torch.double),init_ref_frames.to(torch.double)
        self.trajectory.local_params = init_local_params.to(torch.double)
        if hasattr(self,'res_trajectory'):
            self.res_trajectory._set_frame_attr('origins',init_ori)
            self.res_trajectory._set_frame_attr('ref_frames',init_ref_frames)
            self.res_trajectory._set_frame_attr('local_params',init_local_params)
        
    def _integration_step(self,save_every,integration_mod):
        linker_bp_index = self._get_cur_change_index(integration_mod)
        prot_sl_index = self._apply_rotation(linker_bp_index)
        
        static_origins = torch.vstack([self.trajectory.origins[:linker_bp_index],self.trajectory.origins[prot_sl_index:]])
        e_dif_components,e_mat,s_mat = self.energy.get_energy_dif(self.prev_e,static_origins,self.geom_func.origins[linker_bp_index:prot_sl_index],
                                                                   self.geom_func,linker_bp_index,prot_sl_index)
        
        e_dif_components = torch.hstack(e_dif_components)
    
        Del_E=e_dif_components.sum()

        r = torch.rand(1).item()
        self.total_step += 1

        if not Del_E.isnan() and Del_E < 0 or (not(torch.isinf(torch.exp(Del_E))) and (r  <= torch.exp(-Del_E/self._scaled_KT))):
            self.energy.update_matrices(e_mat,s_mat,linker_bp_index,prot_sl_index)
            self.prev_e += e_dif_components
            self.accepted_steps += 1
            
            self.trajectory.origins = self.geom_func.origins
            self.trajectory.ref_frames = self.geom_func.ref_frames
            self.trajectory.local_params = self.geom_func.local_params
            
            if (self.accepted_steps% save_every) == 0:
                self.energy_comp_traj[self.trajectory.cur_step] = self.prev_e
                self.trajectory.cur_step += 1
                if self.trajectory.cur_step == self.transfer_to_memory_every:
                    self.prev_e = torch.hstack(self.energy.get_energy_components(self.geom_func))
                    self._transfer_to_memory()
                    
                    self.trajectory.cur_step = 0
       
                self.trajectory.origins = self.geom_func.origins
                self.trajectory.ref_frames = self.geom_func.ref_frames
                self.trajectory.local_params = self.geom_func.local_params
                
    def _signal_handler(self,signum, frame):
        self._stop_loop = True
            
    def _apply_rotation(self,linker_bp_index):   
        self.geom_func.ref_frames = self.trajectory.ref_frames.clone()
        self.geom_func.local_params = self.trajectory.local_params.clone()
        self.geom_func.origins = self.trajectory.origins.clone()
        self.geom_func.local_params[linker_bp_index] += torch.normal(mean=self.normal_mean,std=self.normal_scale,out=self.change)
        prot_sl_index = self.trajectory.get_proteins_slice_ind(linker_bp_index)
        self.geom_func.rotate_ref_frames_and_ori(linker_bp_index,prot_sl_index)
        
        
        return prot_sl_index
        
        
    def _get_cur_change_index(self,integration_mod):
        if integration_mod == 'minimize':
            cur_index = self.cur_movable_ind
            self.cur_movable_ind += 1
            if self.cur_movable_ind == self.movable_ind.shape[0]:
                self.cur_movable_ind = 0
        elif integration_mod == 'random_step':
            cur_index = torch.randint(self.movable_ind_len,(1,))
            
        return self.movable_ind[cur_index]
    
    def _transfer_to_memory(self,steps=None):
        if not steps:
            steps = self.transfer_to_memory_every
        
        dna_len = self.trajectory.data_len
        origins_traj = self.trajectory.origins_traj[:steps,:dna_len].numpy(force=True)
        ref_frames_traj = self.trajectory.ref_frames_traj[:steps].numpy(force=True)
        local_params_traj = self.trajectory.local_params_traj[:steps].numpy(force=True)
        energy_comp_traj = self.energy_comp_traj[:steps].numpy(force=True)

        
        for i in range(steps):
            self.res_trajectory.cur_step += 1
            self.res_trajectory.add_frame(self.res_trajectory.cur_step,origins=origins_traj[i],ref_frames = ref_frames_traj[i],
                                         local_params=local_params_traj[i],energies=energy_comp_traj[i])
    
   