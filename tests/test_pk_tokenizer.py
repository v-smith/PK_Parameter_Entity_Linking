import pytest

from pk_el.tokenizers.pk_tokenizer import pk_tokenizer, basic_preprocessing

# test cases
test_cases = [
    # AUCS
    # auc
    ("auc", ["auc"]),
    ("area under curve", ["auc"]),
    ("exposure", ["auc"]),
    ("total exposure", ["auc"]),
    ("systemic exposure", ["auc"]),
    # auclast
    ("auclast", ["auc", "last"]),
    ("aucall", ["auc", "last"]),
    ("auctend", ["auc", "last"]),
    ("auc_last_obs", ["auc", "last"]),
    ("auc_last_meas", ["auc", "last"]),
    ("auc0-last",  ["auc", "last"]),
    ("auc0-tlast",  ["auc", "last"]),
    # auc%ext
    ("auc%ext", ['auc', 'extrapolated', 'percentage']),
    ("auc_extrapolated%", ["auc", "extrapolated", "percentage"]),
    # aucinf
    ("auc∞", ["auc", "inf"]),
    ("auc0-∞", ["auc", "inf"]),
    ("aucinf", ["auc", "inf"]),
    ("auc0-inf", ["auc", "inf"]),
    ("auc to infinity", ["auc", "inf"]),
    # auct
    ("auct", ["auc", "t"]),
    ("aucτ", ["auc", "t"]),
    ("auc0-τ", ["auc", "t"]),
    ("auc0-t", ["auc", "t"]),
    ("interval auc", ["auc", "t"]),
    ("dose interval auc", ["auc", "t"]),
    ("auct", ["auc", "t"]),
    ("auc0-12h", ["auc", "t"]),
    # aumc
    ("aumc", ["aumc"]),
    ("first-moment curve", ["aumc"]),
    ("auc_first_moment", ["aumc"]),
    ("aumc0-t", ["aumc", "t"]),
    ("aumc0-inf", ["aumc", "inf"]),
    # auc ss
    ("steady-state auc", ['auc', 'ss']),
    ("auc steady state", ["auc", "ss"]),
    # auc ratio
    ("auc ratio", ["auc", "ratio"]),
    ("aucr", ["auc", "ratio"]),
    ("auc_dr", ["auc", "ratio"]),
    ("relative auc", ["auc", "ratio"]),
    ("auc interaction ratio", ["auc", "ratio"]),
    ("parent metabolite auc ratio", ["auc", "ratio"]),
    # auc/dose
    ("auc/dose", ["auc", "dosenorm"]),
    ("auc (dn)", ["auc", "dosenorm"]),
    ("dose-normalised auc", ['auc', 'dosenorm']),
    ("auc_per_dose", ["auc", "dosenorm"]),
    ("normalised auc", ['auc', 'dosenorm']),

    #BIOAVs
    # fab
    ('f', ['f']),
    ('fab', ['f']),
    ('absolute bioavailability', ['f']),
    ('fraction absorbed', ['f']),
    ('systemic availability', ['f']),
    ('oral bioavailability', ['f']),
    ('bioavailability', ['f']),
    ('absolute oral bioavailability', ['f']),
    # frel
    ('frel', ['frel']),
    ('relative bioavailability', ['frel']),
    ('relative systemic availability', ['frel']),
    # fg
    ('fg', ['fg']),
    ('gut-wall bioavailability', ['fg']),
    ('gut wall bioavailability', ['fg']),
    # fh
    ('fh', ['fh']),
    ('hepatic bioavailability', ['fh']),
    # fr
    ('fr', ['fr']),
    ('renal bioavailability', ['fr']),

    # CLEARANCES
    # cl
    ('cl', ['cl']),
    ('cltot', ['cl']),
    ('clt', ['cl']),
    ('clearance', ['cl']),
    ('total clearance', ['cl']),
    ('clearance rate', ['cl']),
    ('θCL', ["cl"]),
    # cl ratio
    ('cl_ratio', ['cl', 'ratio']),
    ('clearance ratio', ['cl', 'ratio']),
    # cl bionorm
    ('cl/f', ['bionorm', 'cl']),
    ('clpo', ['bionorm', 'cl']),
    ('clp/f', ['bionorm', 'cl']),
    ('oral clearance', ['bionorm', 'cl']),
    ('apparent oral clearance', ['bionorm', 'cl']),
    ('clz', ['bionorm', 'cl']),
    ('apparent clearance', ['bionorm', 'cl']),
    # clb
    ('clb', ['clb']),
    ('blood clearance', ['clb']),
    ('total blood clearance', ['clb']),
    # clh
    ('clh', ['clh']),
    ('hepatic clearance', ['clh']),
    ('cl(h)', ['clh']),
    ('metabolic clearance', ['cl', 'metabolic']),
    # clr
    ('clr', ['clr']),
    ('renal clearance', ['clr']),
    ('cl(r)', ['clr']),
    # clnr
    ('clnr', ['clnr']),
    ('nonrenal clearance', ['clnr']),
    ('cl(nr)', ['clnr']),
    # clu
    ('clu', ['clu']),
    ('unbound clearance', ['clu']),
    # clint
    ('clint', ['clint']),
    ('intrinsic metabolic clearance', ['clint']),
    ('enzyme mediated clearance', ['clint']),
    ('intrinsic clint', ['clint']),
    # fe
    ('fe', ['fe']),
    ('fraction excreted unchanged', ['fe']),
    ('cumulative urinary excretion', ['fe']),
    ('excretion', ['fe']),
    ('renal excretion', ['fe']),
    # ae
    ('ae', ['ae']),
    ('cumulative excretion amount', ['ae']),
    ('amount excreted', ['ae']),

    # INTER-COMPARTMENTAL CL
    # q
    ('q', ['q']),
    ('q1', ['q']),
    ('qp1', ['q']),
    ('intercompartmental clearance', ['q']),
    ('intercompartmental clearance central peripheral compartment', ['q']),
    ('distribution clearance', ['q']),
    ('1st intercompartmental clearance', ['q']),
    # q2
    ('q2', ['q2']),
    ('qp2', ['q2']),
    ('intercompartment clearance central second peripheral compartment', ['q2']),
    ('2nd intercompartmental clearance', ['q2']),
    # q/f
    ('q/f', ['bionorm', 'q']),
    ('q1/f', ['bionorm', 'q']),
    ('qp1/f', ['bionorm', 'q']),
    ('apparent intercompartmental clearance central peripheral compartment', ['bionorm', 'q']),
    # q2/f
    ('q2/f', ['bionorm', 'q2']),
    ('qp2/f', ['bionorm', 'q2']),
    ('bioavailability normalised apparent intercompartmental clearance central second peripheral compartment', ['bionorm', 'q2']),

    # CONCENTRATIONS
    # cmax
    ('cmax', ['cmax']),
    ('cpeak', ['cmax']),
    ('peak concentration', ['cmax']),
    ('maximum concentration', ['cmax']),
    ('maximum level', ['cmax']),
    ('peak', ['cmax']),
    ('cmax/dose', ['cmax', 'dosenorm']),
    ('cmax (dn)', ['cmax', 'dosenorm']),
    ('dose-normalised cmax', ['cmax', 'dosenorm']),
    ('cmax per dose', ['cmax', 'dosenorm']),
    ('normalised cmax', ['cmax', 'dosenorm']),
    ('cmax_ratio', ['cmax', 'ratio']),
    ('peak concentration ratio', ['cmax', 'ratio']),
    # cmin
    ('cmin', ['cmin']),
    ('minimum observed concentration', ['cmin']),
    # css
    ('css', ['c', 'ss']),
    ('steady-state concentration', ['c', 'ss']),
    # css ratio
    ('css_ratio', ['c', 'ratio', 'ss']),
    ('steady-state concentration ratio', ['c', 'ratio', 'ss']),
    # c trough
    ('ctrough', ['ctrough']),
    ('cpre', ['ctrough']),
    ('pre-dose level', ['ctrough']),
    ('trough concentration', ['ctrough']),
    ('trough level', ['ctrough']),
    ('trough', ['ctrough']),
    # c0
    ('c0', ['c0']),
    ('extrapolated concentration', ['c0']),
    # cavg
    ('cavg', ['cavg']),
    ('cav', ['cavg']),
    ('cave', ['cavg']),
    ('caverage', ['cavg']),
    ('average concentration', ['cavg']),
    # ct
    ('ct', ['c', 't']),
    ('cd30', ['c', 't']),
    ('c7', ['c','t']),

    # Half-lives
    # t1/2
    ('t1/2z', ['t1/2']),
    ('t1/2', ['t1/2']),
    ('terminal t1/2', ['t1/2']),
    ('elimination t1/2', ['t1/2']),
    ('t1/2 elimination t1/2el', ['t1/2']),
    ('t1/2λz', ['t1/2']),
    ('k10 t1/2', ['t1/2']),
    ('apparent terminal t1/2', ['t1/2']),
    ('t1/2app', ['t1/2']),
    ('t1/2ke', ['t1/2']),
    ('terminal elimination t1/2', ['t1/2']),
    ('effective t1/2', ['t1/2']),
    # t1/2 ratio,
    ('t1/2_ratio', ['ratio', 't1/2']),
    # t1/2 alpha
    ('t1/2,α', ['alpha', 't1/2']),
    ('distribution t1/2', ['alpha', 't1/2']),
    ('fast t1/2', ['alpha', 't1/2']),
    ('t1/2 alpha', ['alpha', 't1/2']),
    ('initial t1/2', ['alpha', 't1/2']),
    # t1/2 beta
    ('t1/2,β', ['beta', 't1/2']),
    ('secondary t1/2', ['beta', 't1/2']),
    ('2nd t1/2', ['beta', 't1/2']),
    ('slow t1/2', ['beta', 't1/2']),
    ('t1/2beta', ['beta', 't1/2']),
    ("elimination half life calculated from the beta-slope", ['beta', 't1/2']),
    # t1/2 gamma
    ('t1/2,γ', ['gamma', 't1/2']),
    ('t1/2 gamma', ['gamma', 't1/2']),
    ('3rd t1/2', ['gamma', 't1/2']),
    # t1/2 ka
    ('t1/2,ka', ['kabs', 't1/2']),
    ('t1/2abs', ['kabs', 't1/2']),
    ('t1/2ab', ['kabs', 't1/2']),
    ('absorption t1/2', ['kabs', 't1/2']),
    ('k01 t1/2', ['kabs', 't1/2']),

    # Rate constants
    # kel
    ('kel', ['kel']),
    ('λ', ['kel']),
    ('λz', ['kel']),
    ('k', ['k']),
    ('ke', ['kel']),
    ('kelimination', ['kel']),
    ('kout', ['kout']),
    ('k10', ['kel']),
    ('elimination rate', ['kel']),
    ('first-order elimination rate', ['kel']),
    ('elimination rate constant', ['kel']),
    # kcomp
    ('kcomp', ['kcomp']),
    ('k12', ['kcomp']),
    ('k21', ['kcomp']),
    ('distribution rate constant', ['kcomp']),
    ('transfer rate constant', ['kcomp']),
    ('transfer rate constant central to peripheral', ['kcomp']),
    ('transfer rate constant peripheral to central', ['kcomp']),
    # kabs,
    ('kabs', ['kabs']),
    ('k01', ['kabs']),
    ('ka', ['kabs']),
    ('kab', ['kabs']),
    ('absorption rate constant', ['kabs']),
    ('absorption rate', ['kabs']),
    ('first-order absorption rate', ['kabs']),
    # kexcr
    ('kexcr', ['kexcr']),
    ('urinary excretion rate constant', ['kexcr']),
    ('k_excr', ['kexcr']),
    ('kexcretion', ['kexcr']),
    # kmet
    ('kmet', ['kmet']),
    ('kmetabolite', ['kmet']),
    ('metabolite elimination rate constant', ['kmet']),
    # lambdas
    ('λ1', ['1', 'lambda']),
    ('alpha', ['alpha']),
    ('λ2', ['2', 'lambda']),
    ('beta', ['beta']),

    # TIMES
    #tmax
    ('tmax', ['tmax']),
    ('tpeak', ['tmax']),
    ('peak time', ['tmax']),
    ('time to maximum concentration', ['tmax']),
    ('time to cmax', ['tmax']),
    ('time to peak concentration', ['tmax']),
    ('maximum time to cmax', ['tmax']),
    # tmax ratio
    ('tmax_ratio', ['ratio', 'tmax']),
    # tlag
    ('tlag', ['tlag']),
    ('alag', ['tlag']),
    ('absorption lag time', ['tlag']),
    # tss
    ('tss', ['ss', 't']),
    ('time to steady state', ['ss', 't']),
    # mrt
    ('mrt', ['mrt']),
    ('mean residence time', ['mrt']),
    ('mrt0-t', ['mrt']),
    ('mrt0-inf', ['mrt']),
    ('mrtinf', ['inf', 'mrt']),
    ('mrtlast', ['last', 'mrt']),
    # mat
    ('mat', ['mat']),
    ('mean absorption time', ['mat']),
    # mtt
    ('mtt', ['mtt']),
    ('mean transit time', ['mtt']),

    # Volumes
    # v
    ('vd', ['v']),
    ('v', ['v']),
    ('vz', ['v']),
    ('distribution volume', ['v']),
    ('volume distribution', ['v']),
    ('volume', ['v']),
    # v1
    ('vc', ['v1']),
    ('v1', ['v1']),
    ('central volume', ['v1']),
    ('central v', ['v1']),
    # v2
    ('vp', ['v2']),
    ('vp1', ['v2']),
    ('v2', ['v2']),
    ('peripheral volume', ['v2']),
    ('peripheral vd', ['v2']),
    ('v beta', ['v']),
    # v3
    ('vp2', ['v3']),
    ('v3', ['v3']),
    ('second peripheral volume', ['v3']),
    ('second peripheral vd', ['v3']),
    ('v gamma', ['v']),
    # v/f
    ('vd/f', ['bionorm', 'v']),
    ('vz/f', ['bionorm', 'v']),
    ('v/f', ['bionorm', 'v']),
    ('apparent vd', ['bionorm', 'v']),
    ('apparent volume', ['bionorm', 'v']),
    # v1/f
    ('vc/f', ['bionorm', 'v1']),
    ('v1/f', ['bionorm', 'v1']),
    ('apparent central vd', ['bionorm', 'v1']),
    # v2/f
    ('vp/f', ['bionorm', 'v2']),
    ('v2/f', ['bionorm', 'v2']),
    ('apparent peripheral vd', ['bionorm', 'v2']),
    # v3/f
    ('vp2/f', ['bionorm', 'v3']),
    ('v3/f', ['bionorm', 'v3']),
    ('apparent second peripheral vd', ['bionorm', 'v3']),
    # vss
    ('steady state volume', ['ss', 'v']),
    ('steady state vd', ['ss', 'v']),

    # METAB/EXCR
    # vmax
    ('vmax', ['vmax']),
    ('maximum enzyme activity', ['vmax']),
    # km
    ('km', ['km']),
    ('michaelis menten constant', ['km']),
    # e
    ('e', ['e']),
    ('extraction ratio', ['e']),
    # eh
    ('eh', ['eh']),
    ('hepatic extraction ratio', ['eh']),
    # fm
    ('fm', ['fm']),
    ('metabolic conversion fraction', ['fm']),
    # fu
    ('fu', ['fu']),
    ('fraction unbound', ['fu']),
    ('fraction protein unbound', ['fu']),
    ('free fraction', ['fu']),
    ('protein binding ratio', ['fu']),
    ('protein binding', ['fu']),
    # blood flow
    ('blood flow', ['blood', 'flow']),
]

@pytest.mark.parametrize("input_text, expected_output", test_cases)
def test_pk_tokenizer(input_text, expected_output):
    """Test pk_tokenizer for correct tokenization."""
    input_text = basic_preprocessing(input_text)
    result = pk_tokenizer(input_text)
    assert result == expected_output, f"Failed on input: {input_text}. Expected: {expected_output}, Got: {result}"

if __name__ == "__main__":
    pytest.main()