import ast
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time

repo=Path('/Users/dot/workspace/fp201')
probe_dir=Path(tempfile.mkdtemp(prefix='fp201-nbi-scan-',dir='/private/tmp'))
os.chdir(probe_dir)
module=ast.parse((repo/'FPQL_201.py').read_text())
module.body=[node for node in module.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef))]
namespace={'__name__':'fp201_probe'}
exec(compile(module,str(repo/'FPQL_201.py'),'exec'),namespace)
np=namespace['np']
solve_banded=namespace['solve_banded']
started=time.monotonic()
with (probe_dir/'initialization.log').open('w') as log, contextlib.redirect_stdout(log):
    ctx,ssp,out,plot,para=namespace['read_config_file'](str(repo/'int_1.json'))
    namespace['ctx']=ctx
    namespace['ipsii']=[]
    namespace['apply_config_settings'](ssp,para)
    ssp.nbi_enabled=1
    ssp.icrh_enabled=0
    ssp.nrdpsi=1
    ssp.nphi=ctx.nphi
    ssp.ipsii=[1]
    ssp.ipsiii=[1]
    ssp.info=1
    ssp.arrays=1
    ssp.positions=np.array([1.],dtype=float)
    ssp.joutql=1
    ssp.current_out_dir=str(probe_dir)
    ssp.output_paths={key:str(probe_dir/(key+'.txt')) for key in ['read','case','qleg00','A_full','A_banded','b_vector','A_out','b_out','Pn','Fv','Fi','name']}
    ssp.inifpq()
    ssp.coelgd_with_weight_optimized(enable_weight=bool(ssp.enable_trapping),uc=ssp.trapping_ucr,a=ssp.trapping_a)
    ssp.iniqln(ssp.ispec,0)
    ssp.inimat(0)
    ssp.inicol(ssp.ispec,1)
    ssp.iniqln(ssp.ispec,1)
    ssp.inimat(1)
    ssp.matrix()
    A=ssp.fmatrx[ssp.iwd:3*ssp.iwd+1,:].copy()
    b1=ssp.fb.copy()
    b0=np.zeros_like(b1)
    b0[0]=ssp.fb1_mxw if ssp.fb1_prv==0 else ssp.fb1_prv
    sig1=ssp.sig.copy()
    # Verify operator coefficients do not change when changing source amplitude.
    op_diffs=[]
    for ip in [0,33,122,199]:
        for gauss_index in range(ssp.ngaus):
            ssp.sig=sig1
            ssp.mastfp(ip,gauss_index)
            gamma1=ssp.fgamm.copy()
            ssp.sig=10*sig1
            ssp.mastfp(ip,gauss_index)
            op_diffs.append(float(np.max(np.abs(ssp.fgamm-gamma1))))
    ssp.sig=sig1

    energies=ssp.tbkev.copy()
    energy_mask=np.abs(energies-90)<=4.5
    rows=[]
    for power in [.1,.5,1.,2.,5.,10.]:
        rhs=b0+power*(b1-b0)
        raw=solve_banded((ssp.iwd,ssp.iwd),A,rhs)
        coefficients=raw.reshape(ssp.nptu,2*ssp.nfvrb)[:,0::2]
        raw_f=coefficients@ssp.plgcmu
        raw_window=raw_f[energy_mask,:]
        raw_peak=float(np.max(raw_window))
        fnorm=float(4*np.pi*ssp.qlmomu(coefficients[:,0],ssp.du,ssp.nptu,0))
        ssp.fb=raw.copy()
        ssp.disfun(1)
        assert np.isclose(fnorm,ssp.fnorm,rtol=1e-14)
        fnorm=float(ssp.fnorm)
        normalized_f=ssp.tbfu@ssp.plgcmu
        window=normalized_f[energy_mask,:]
        i_energy,i_mu=np.unravel_index(np.argmax(window),window.shape)
        actual_energy=energies[np.flatnonzero(energy_mask)[i_energy]]
        rows.append({'power_Wcm3':power,'raw_peak':raw_peak,'fnorm':fnorm,'normalized_peak':float(np.max(window)),'peak_energy_keV':float(actual_energy),'peak_mu':float(ssp.xmu[i_mu]),'f0_boundary':float(raw[0])})
        np.savez_compressed(probe_dir/f'P-{power}.npz',raw=raw,raw_distribution=raw_f,normalized_distribution=normalized_f,energies=energies,mu=ssp.xmu)

result={'probe_dir':str(probe_dir),'seconds':time.monotonic()-started,'nptu':ssp.nptu,'legmod':ssp.legmod,'nmu':ssp.nmu,'emax_keV':float(ssp.tbkev[-1]),'Ti':ssp.ctx.tempi,'dqlw':float(ssp.dqlw),'background_source_max':float(np.max(np.abs(ssp.tab_source))),'fb1_mxw':float(ssp.fb1_mxw),'operator_coefficient_maxdiff_P1_vs_P10':max(op_diffs),'rows':rows}
(probe_dir/'scan.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
