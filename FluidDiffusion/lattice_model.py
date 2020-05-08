import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# software constants
IMAGE_MAXIMUM_INTENSITY = 255

# physical parameters; all constants are in S.I. units
FLUID_ELEMENT_SIZE = 10 ** -6
TEMPERATURE_PREFACTOR = 0.0000167   # defined as e^2 / 4 pi epsilon k_B
MASS_FACTOR = 0.01097               # defined as sqrt(m_u / k_B)
CHARACTERISTIC_TEMPERATURE = TEMPERATURE_PREFACTOR / FLUID_ELEMENT_SIZE
ROOM_TEMPERATURE = 298
TIME_STEPS = 10


def get_time_step(temperature, molecular_mass, degrees_of_freedom):
    return (MASS_FACTOR * FLUID_ELEMENT_SIZE * molecular_mass) / \
        np.sqrt(degrees_of_freedom * temperature)


def get_total(field):
    def get_total_charge(cell):
        return cell.population

    vectorized_get_total_charge = np.vectorize(get_total_charge)
    return vectorized_get_total_charge(field)


class ChargeStack:
    def __init__(self, positive_occupancy, negative_occupancy):
        self.charges = []
        if positive_occupancy > 0:
            self.charges = self.charges + list(
                np.random.choice([1, -1], int(positive_occupancy))
            )
        if negative_occupancy > 0:
            self.charges = self.charges + list(
                np.random.choice([1, -1], int(positive_occupancy))
            )
        else:
            self.charges = []

    def total_charge(self):
        return sum(self.charges)

    def population(self):
        return len(self.charges)


class Medium:
    def __init__(
        self,
        positive_particles,
        negative_particles,
        relative_permitivity=1
    ):
        """
        Generates a fluid medium with certain charge distributions from two,
        2D numpy arrays, representing the spatial distribution of positive and
        negative particles.
        """
        vectorized_charge_stack = np.vectorize(ChargeStack)
        self.particle_field = vectorized_charge_stack(
            positive_particles,
            negative_particles
        )
        self.relative_permitivity = relative_permitivity

    def update_charge_distribution(self, temperature=ROOM_TEMPERATURE):
        self_field = self.particle_field
        w, h = self_field.shape
        mesh_x, mesh_y = np.meshgrid(
            np.arange(0, w, 1),
            np.arange(0, h, 1)
        )
        vectorized_charge_stack = np.vectorize(ChargeStack)
        template = vectorized_charge_stack(np.zeros((w, h)), np.zeros((w, h)))

        def move_cell(x, y):
            def move_particle(x, y, q1):
                def get_boltzmann_factor(new_x, new_y):
                    if new_x not in [-1, w] and new_y not in [-1, h]:
                        q2 = self_field[new_x, new_y].total_charge()
                        return np.exp(
                            -(CHARACTERISTIC_TEMPERATURE * q1 * q2) /
                            (self.relative_permitivity * temperature)
                        )
                    else:
                        return 0

                bf_left = get_boltzmann_factor(x - 1, y)
                bf_right = get_boltzmann_factor(x + 1, y)
                bf_top = get_boltzmann_factor(x, y - 1)
                bf_bottom = get_boltzmann_factor(x, y + 1)
                # turn them into normalized probabilities
                bf_sum = bf_left + bf_right + bf_top + bf_bottom
                # turn them into cumulative range limits so that we can use
                # Monte-Carlo to randomly "walk" particles
                limit_right = bf_left + bf_right
                limit_top = limit_right + bf_top
                # generate a random number
                random_number = bf_sum * np.random.random()
                if random_number < bf_left:
                    template[x - 1, y].charges.append(q1)
                elif bf_left <= random_number < limit_right:
                    template[x + 1, y].charges.append(q1)
                elif limit_right <= random_number < limit_top:
                    template[x, y - 1].charges.append(q1)
                elif limit_top <= random_number < bf_sum:
                    template[x, y + 1].charges.append(q1)

            charge_list = np.array(self_field[x, y].charges.copy())
            if len(charge_list):
                for charge in np.array(self_field[x, y].charges.copy()):
                    move_particle(x, y, charge)

        vectorized_move_cell = np.vectorize(move_cell)
        vectorized_move_cell(mesh_x, mesh_y)
        self.particle_field = template

    def get_charge_field(self):
        def get_total_charge(cell):
            return cell.total_charge()
        vectorized_get_total_charge = np.vectorize(get_total_charge)
        return vectorized_get_total_charge(self.particle_field)

    def get_population_field(self):
        def get_total_charge(cell):
            return cell.population()
        vectorized_get_total_charge = np.vectorize(get_total_charge)
        return vectorized_get_total_charge(self.particle_field)

    def to_image(
        self,
        intensity_factor=1,
        image_rescaling_factor=1,
        mode="population"
    ):
        if mode == "population":
            population_field = self.get_population_field()
            w, h = image_rescaling_factor * np.array(population_field.shape)
            return Image.fromarray(
                intensity_factor * population_field.transpose() - 1
            ).resize((w, h))
        elif mode == "charge":
            charge_field = self.get_charge_field()
            w, h = image_rescaling_factor * np.array(charge_field.shape)
            return Image.fromarray(
                intensity_factor * charge_field.transpose() - 1
            ).resize((w, h))
        else:
            raise Exception(f"{mode} is not a valid mode!")


def charge_maps_from_image(positive, negative, image_scaling_factor=1):
    """
    Opens two image files which graphically maps the charge distribution in the
    medium, or PIL Image objects, then convert them into numpy arrays. Change
    the image_scaling_factor to change the charge value i.e. if a pixel (fluid
    element) has intensity 100, for image_scaling_factor of 0.2 we get a charge
    of 100 x 0.2 = 20.
    """
    if isinstance(positive, (Path, str)):
        positive = Image.open(positive).convert("L")
    if isinstance(negative, (Path, str)):
        negative = Image.open(negative).convert("L")
    return (
        image_scaling_factor * np.array(positive, dtype="int32")
    ).astype("int32"), (
        image_scaling_factor * np.array(negative, dtype="int32")
    ).astype("int32")


def generate_video(images, video_file_path):
    width, height = images[0].size
    video = cv2.VideoWriter(video_file_path, 0, 1, (width, height))
    for image in images:
        cv_image = np.array(image.convert('RGB'))
        # Convert RGB to BGR
        cv_image = cv_image[:, :, ::-1].copy()
        video.write(cv_image)
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    salt_cube_directory = Path("charge_maps").resolve() / "salt_cube"
    positive_map, negative_map = charge_maps_from_image(
        salt_cube_directory / "salt_cube_positive.png",
        salt_cube_directory / "salt_cube_negative.png",
    )
    movie_images = []
    medium = Medium(positive_map, negative_map)
    img = medium.to_image(5, 1)
    movie_images.append(img)
    for _ in range(TIME_STEPS):
        medium.update_charge_distribution()
        img = medium.to_image(5, 1)
        movie_images.append(img)
    generate_video(movie_images, "diffusion.avi")
