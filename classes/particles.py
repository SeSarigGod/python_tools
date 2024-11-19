from vectors import *
import scipy.constants as const


class Particle3D(object):
    def __init__(self, mass,
                 position,
                 velocity,
                 acceleration,
                 gravity: bool = True, drag: bool = False, spring: bool = False,
                 universal_gravity: bool = True, gravity_acceleration: Vector3 = Vector3((0, -const.g, 0)), gravitational_constant: float = const.G,
                 drag_coefficient: float = 0.1, drag_degree: float = 2.0,
                 spring_constant: float = 1.0, equilibrium: Vector3 = Vector3(),
                 vector_logging: bool = False) -> None:
        self.mass = mass
        self.position = Vector3(position)
        self.velocity = Vector3(velocity)
        self.acceleration = Vector3(acceleration)
        self.bool_gravity = gravity
        self.bool_drag = drag
        self.bool_spring = spring
        self.vector_logging = vector_logging

        if self.bool_gravity:
            self.gravity_type = universal_gravity
            self.gravitational_constant = gravitational_constant
            self.gravity_acceleration = gravity_acceleration

        if self.bool_drag:
            self.drag_coefficient = drag_coefficient
            self.drag_degree = drag_degree

        if self.bool_spring:
            self.spring_constant = spring_constant
            self.equilibrium = equilibrium

        if self.vector_logging:
            self.logged_position = [self.position.to_list()]
            self.logged_velocity = [self.velocity.to_list()]
            self.logged_acceleration = [self.acceleration.to_list()]

    def __repr__(self) -> str:
        return (f"Particle(mass: {self.mass}, position: {self.position}, "
                f"velocity: {self.velocity}, acceleration: {self.acceleration})")

    def __get__(self, instance, owner) -> 'Particle3D':
        return self

    def __set__(self, instance, value: 'Particle3D') -> None:
        self.mass = value.mass
        self.position = value.position
        self.velocity = value.velocity
        self.acceleration = value.acceleration

    def __delete__(self, instance) -> None:
        del self

    def __call__(self, _mass, _position, _velocity, _acceleration) -> None:
        self.mass = _mass
        self.position = _position
        self.velocity = _velocity
        self.acceleration = _acceleration

    def __eq__(self, other: 'Particle3D') -> bool:
        return (self.mass == other.mass and
                self.position == other.position and
                self.velocity == other.velocity and
                self.acceleration == other.acceleration)

    def __ne__(self, other: 'Particle3D') -> bool:
        return (self.mass != other.mass or
                self.position != other.position or
                self.velocity != other.velocity or
                self.acceleration != other.acceleration)

    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)

    def __getattr__(self, key) -> any:
        return super().__getattribute__(key)

    def __delattr__(self, key) -> None:
        super().__delattr__(key)

    def __dict__(self) -> dict:
        return {'mass': self.mass,
                'position': self.position,
                'velocity': self.velocity,
                'acceleration': self.acceleration}

    def set(self, key: str, value) -> 'Particle3D':
        if key == 'mass':
            self.mass = value
        elif key == 'position':
            self.position.set('vector', value)
        elif key == 'velocity':
            self.velocity.set('vector', value)
        elif key == 'acceleration':
            self.acceleration.set('vector', value)
        else:
            raise AttributeError(f"Attribute {key} not found.")
        return self

    def get(self, key: str) -> any:
        if key == 'mass':
            return self.mass
        elif key == 'position':
            return self.position
        elif key == 'velocity':
            return self.velocity
        elif key == 'acceleration':
            return self.acceleration
        else:
            raise AttributeError(f"Attribute {key} not found.")

    def universal_gravity(self, others: 'Particle3D' or list['Particle3D']) -> Vector3:
        if isinstance(others, Particle3D):
            return (const.G * self.mass * others.mass / (self.position - others.position).magnitude()**3) * (self.position - others.position)
        else:
            total_force = Vector3()
            for other in others:
                total_force += (const.G * self.mass * other.mass / (self.position - other.position).magnitude()**3) * (self.position - other.position)
            return total_force

    def planet_gravity(self) -> Vector3:
        return self.mass * self.gravity_acceleration

    def spring(self, k: float, origin: Vector3 = Vector3()) -> Vector3:
        return -k * (self.position - origin)

    def drag(self, coefficient: float, degree: float = 2.0) -> Vector3:
        return -coefficient * (self.velocity.magnitude()**(degree-1)) * self.velocity

    def update_mass(self, mass: float) -> None:
        self.mass = mass

    def update_acceleration(self, outside_force: Vector3 or not str, other_particles: 'Particle3D' or list['Particle3D'] = None) -> None:
        total_force = Vector3()

        if outside_force is not None:
            total_force += outside_force

        if self.bool_gravity:
            if self.gravity_type:
                total_force += self.planet_gravity()
            elif other_particles:
                total_force += self.universal_gravity(other_particles)
            else:
                raise ValueError("No gravity type selected.")

        if self.bool_drag:
            total_force += self.drag(self.drag_coefficient, self.drag_degree)

        if self.bool_spring:
            total_force += self.spring(self.spring_constant, self.equilibrium)

        self.acceleration = total_force / self.mass
        if self.vector_logging:
            self.logged_acceleration.append(self.acceleration.to_list())

    def update_velocity(self, step_size: float) -> None:
        self.velocity += self.acceleration * step_size
        if self.vector_logging:
            self.logged_velocity.append(self.velocity.to_list())

    def update_position(self, step_size: float) -> None:
        self.position += self.velocity * step_size + self.acceleration * 0.5 * step_size**2
        if self.vector_logging:
            self.logged_position.append(self.position.to_list())

    def update(self, step_size: float, outisde_force: Vector3 or not str = None, other_particles: 'Particle3D' or list['Particle3D'] = None) -> 'Particle3D':
        self.update_acceleration(outisde_force, other_particles)
        self.update_velocity(step_size)
        self.update_position(step_size)
        return self
