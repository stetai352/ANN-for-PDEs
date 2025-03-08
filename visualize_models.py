"""
### Visualize solutions

mu = parameter_space.sample_randomly()

U = fom.solve(mu)
U_ann_red = ann_rom.solve(mu)
U_ann_red_recon = ann_reductor.reconstruct(U_ann_red)
U_rb_red = rb_rom.solve(mu)
U_rb_red_recon = rb_reductor.reconstruct(U_rb_red)

fom.visualize(
    (U, U_ann_red_recon, U_rb_red_recon),
    legend = (
        f"Full solution for parameter {mu}",
        f"Reduced ANN solution for parameter {mu}",
        f"Reduced RB solution for parameter {mu}",
    ),
)

### Error Testing
#mu = [0.1,1.,0.1,1]
mu = parameter_space.sample_randomly()
U = fom.solve(mu)
U_ann_red = ann_rom.solve(mu)
U_ann_red_recon = ann_reductor.reconstruct(U_ann_red)
U_rb_red = rb_rom.solve(mu)
U_rb_red_recon = rb_reductor.reconstruct(U_rb_red)

fom.visualize(
    (U - U_ann_red_recon, U - U_rb_red_recon),
    legend = (
        f"Reduced ANN error {mu}",
        f"Reduced RB error {mu}",
    ),
)
"""