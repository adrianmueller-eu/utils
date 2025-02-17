import numpy as np
import matplotlib.pyplot as plt

def quadratic_effect(observations, naive_expectation_fn, n_samples=10**6):
    """
    Simulates how observations are biased when the probability of observing something
    scales (linearly) with its availability in the system.

    Args:
        observations (list): numerical observations, non-negative and normalizable
        naive_expectation_fn (function): naively expected experience with a given observation (without the effect)

    The average experience will be:
        E[availability * naive expectation] / E[availability]
    """
    p_observed = observations / observations.sum()
    observed = np.random.choice(observations, size=n_samples, p=p_observed)
    # print(f"Theoretical average: {(observations * naive_expectation_fn(observations)).mean() / observations.mean():.2f}")
    return naive_expectation_fn(observed).mean()

    # Example: Bus waiting time
    bus_interval = 20
    noise_std = 5
    n_buses = 10000
    arrival_times = np.arange(n_buses) * bus_interval + np.random.normal(0, noise_std, n_buses)
    intervals = np.maximum(0, np.diff(np.sort(arrival_times)))
    avg_wait = quadratic_effect(intervals, naive_expectation_fn=lambda x: x/2)  # I'd naively expect to wait on average half the regular interval
    print(f"Average experienced wait: {avg_wait:.2f} min")

    # Exmaple: Friendship paradox
    avg_friends = 100
    friends_friends = np.random.lognormal(mean=np.log(avg_friends) - 0.5**2/2, sigma=0.5, size=10000)
    avg_friends_friends = quadratic_effect(friends_friends, naive_expectation_fn=lambda x: x)  # I'd naively expect my friends on average to have just as many friends as the average person
    print(f"Average number of friends' friends: {avg_friends_friends:.2f}")

def christmas_tree():
    # after https://community.wolfram.com/groups/-/m/t/175891

    from matplotlib.animation import FuncAnimation
    from colorsys import hsv_to_rgb

    def create_points(t_range, f, cl, sg, hf, dp):
        points = []
        colors = []
        for t in t_range:
            s_val = t**0.6 - f
            x = -sg * s_val * np.sin(s_val)
            y = -sg * s_val * np.cos(s_val)
            z = dp + s_val
            points.append([x, y, -z])

            brightness = 0.6 + sg * 0.4 * np.sin(hf * s_val)
            rgb_color = hsv_to_rgb(cl, 1, brightness)
            colors.append(np.append(rgb_color, 1))
        return np.array(points), np.array(colors)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=180)
    ax.set_box_aspect([1, 1, 1.3])
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-22, 0)
    ax.set_axis_off()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    scatters = [ax.scatter([], [], [], s=1) for _ in range(4)]

    def animate(f):
        t_range = np.arange(0, 200, 0.5)
        for scatter, cl, sg, hf, dp in zip(scatters,
                [1, 0.45, 1, 0.45], [-1, 1, -1, 1], [1, 1, 4, 4], [0, 0, 0.2, 0.2]
            ):
            points, colors = create_points(t_range, f, cl, sg, hf, dp)
            scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
            scatter.set_color(colors)
        return scatters

    frames = np.arange(0, 1, 0.01) % 0.20
    return FuncAnimation(fig, animate, frames=frames, interval=0, blit=False, repeat=True)