Accuracy / ECE on the test sets with calibrators fitted on dev only (mean over slices).

system                 suite                 raw   per-task (dev)   default (dev)  per-task fallbacks
untuned                jevbench    0.742 / 0.239    0.775 / 0.084   0.764 / 0.111  ['prompt_injections']
untuned                custom      0.940 / 0.060    0.951 / 0.031   0.940 / 0.080  ['state_agent_next_tool', 'state_agent_loop']
pilot 2                jevbench    0.793 / 0.076    0.806 / 0.067   0.797 / 0.074  ['prompt_injections']
pilot 2                custom      0.925 / 0.073    0.953 / 0.052   0.929 / 0.088  ['state_agent_next_tool', 'state_agent_loop']
option A (step 500)    jevbench    0.792 / 0.081    0.800 / 0.061   0.786 / 0.083  ['prompt_injections']
option A (step 500)    custom      0.948 / 0.063    0.972 / 0.039   0.943 / 0.069  ['state_agent_next_tool', 'state_agent_loop']
untuned + <bos>        jevbench    0.770 / 0.218    0.799 / 0.088   0.787 / 0.103  ['prompt_injections']
untuned + <bos>        custom      0.948 / 0.052    0.960 / 0.030   0.955 / 0.081  ['state_agent_next_tool', 'state_agent_loop']
<bos> fine-tune (step 1500) jevbench    0.804 / 0.083    0.817 / 0.065   0.808 / 0.076  ['prompt_injections']
<bos> fine-tune (step 1500) custom      0.971 / 0.033    0.981 / 0.029   0.966 / 0.039  ['state_agent_next_tool', 'state_agent_loop']
released s1, int4 (Ollama) jevbench    0.800 / 0.074    0.804 / 0.059   0.802 / 0.070  ['prompt_injections']
released s1, int4 (Ollama) custom      0.964 / 0.043    0.981 / 0.034   0.960 / 0.048  ['state_agent_next_tool', 'state_agent_loop']
