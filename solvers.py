import torch


def euler(model, x_0, NFE: int = 100):
    model.eval()

    x_t = x_0

    bs = x_0.shape[0]

    timesteps = torch.linspace(0, 1, NFE + 1).to(x_t.device)
    h = 1 / NFE

    for t in timesteps:
        t = t.unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t)

        x_t = x_t + h * k1

    return x_t


def midpoint(model, x_0, NFE: int = 100):
    model.eval()

    x_t = x_0

    bs = x_0.shape[0]

    timesteps = torch.linspace(0, 1, NFE + 1).to(x_t.device)
    h = 1 / NFE

    for t in timesteps:
        t = t.unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t)
            k2 = model(x_t + (h / 2) * k1, t + h / 2)

        x_t = x_t + h * k2

    return x_t


def heun(model, x_0, NFE: int = 100):
    model.eval()

    x_t = x_0

    bs = x_0.shape[0]

    timesteps = torch.linspace(0, 1, NFE + 1).to(x_t.device)
    h = 1 / NFE

    for t in timesteps:
        t = t.unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t)
            k2 = model(x_t + h * k1, t + h)

        x_t = x_t + (h / 2) * (k1 + k2)

    return x_t


def rk4(model, x_0, NFE: int = 100):
    model.eval()

    x_t = x_0

    bs = x_0.shape[0]

    timesteps = torch.linspace(0, 1, NFE + 1).to(x_t.device)
    h = 1 / NFE

    for t in timesteps:
        t = t.unsqueeze(0).repeat((bs,))

        with torch.no_grad():
            k1 = model(x_t, t)
            k2 = model(x_t + (h / 2) * k1, t + h / 2)
            k3 = model(x_t + (h / 2) * k2, t + h / 2)
            k4 = model(x_t + h * k3, t + h)

        x_t = x_t + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_t

