from contextlib import contextmanager
import numpy as np
import pandas as pd


class XMLWriter:
    def __init__(self):
        self._doc = []                          # 最后用\n join每一行
        self._level = 0                         # 控制缩进

    def __getattr__(self, name):                # 所有红色的tag
        if name.startswith('_') or name in ("write", "single", "source", "attr"):
            return super(XMLWriter, self).__getattr__(name)
        else:
            def func(**kwargs):
                attrs = []
                for k, v in kwargs.items():
                    attrs.append(' {k}="{v}"'.format(k=k, v=v))
                attrs = "".join(attrs)
                self._level += 1
                self._doc.append("{}<{}{}>".format(" " * (4 * self._level), name, attrs))
                yield
                self._level -= 1
                self._doc.append("{}</{}>".format(" " * (4*self._level), name))
            return contextmanager(func)

    def write(self, text):
        self._level.append(" " * (4*self._level) + str(text))

    def single(self, key, value):
        self._doc.append("{tabs}<{key}>{value}</{key}>".format(
            tabs=" " * (4*self._level),
            key=key,
            value=value,
        ))

    def source(self):
        return "\n".join(self._doc)

    def attr(self, key, **kwargs):
        attrs = []
        for k, v in kwargs.items():
            attrs.append(' {k}="{v}"'.format(k=k, v=v))
        attrs = "".join(attrs)
        self._doc.append("{tabs}<{key}{attrs}/>".format(tabs=" " * (4 * self._level), key=key, attrs=attrs))


class XMLObject:
    def serialize(self):
        raise NotImplementedError


class VehicleType(XMLObject):
    def __init__(self, id, capacity, costs):
        self.id = id
        self.capacity = capacity
        self.costs = costs

    def serialize(self, writer=None):
        w = writer or XMLWriter()
        with w.type():
            w.single("id", self.id)
            w.single("capacity", self.capacity)
            with w.costs():
                for k, v in self.costs.items():
                    w.single(k, v)


class Vehicle(XMLObject):
    def __init__(self, id, type_id, start, end, time_schedule):
        self.id = id
        self.type_id = type_id
        self.start = start
        self.end = end
        self.time_schedule = time_schedule

    def serialize(self, writer=None):
        w = writer or XMLWriter()
        with w.vehicle():
            w.single("id", self.id)
            w.single("typeId", self.type_id)
            with w.startLocation():
                w.single("id", 0)
                w.attr("coord", x=self.start[0], y=self.start[1])
            with w.endLocation():
                w.single("id", 101)
                w.attr("coord", x=self.end[0], y=self.end[1])
            with w.timeSchedule():
                w.single("start", self.time_schedule[0])
                w.single("end", self.time_schedule[1])


class Service(XMLWriter):
    def __init__(self,
                 id: str,
                 type: str,
                 location_id: str,
                 coord: tuple,
                 capacity_demand: float,
                 duration: float,
                 time_windows: list):
        self.id = id
        self.type = type
        self.location_id = location_id
        self.coord = coord
        self.capacity_demand = capacity_demand
        self.duration = duration
        self.time_windows = time_windows

    def serialize(self, writer=None):
        w = writer or XMLWriter()
        with w.service(id=self.id, type=self.type):
            w.single("locationId", self.location_id)
            w.attr("coord", x=self.coord[0], y=self.coord[1])
            w.single("capacity-demand", self.capacity_demand)
            w.single("duration", self.duration)
            with w.timeWindows():
                for window in self.time_windows:
                    with w.timeWindow():
                        w.single("start", window[0])
                        w.single("end", window[1])


class Problem:
    def __init__(self):
        self.vehicles = []
        self.vehicle_types = []
        self.services = []

    def add_vehicle_type(self, vehicle_type):
        self.vehicle_types.append(vehicle_type)

    def add_vehile(self, vehicle):
        self.vehicles.append(vehicle)

    def add_service(self, service):
        self.services.append(service)

    def to_xml(self, filename):
        w = XMLWriter()
        with w.problem(**{"xmlns": "http://www.w3schools.com",
                          "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                          "xsi:schemaLocation": "http://www.w3schools.com vrp_xml_schema.xsd"}):
            with w.problemType():
                w.single("fleetSize", "INFINITE")
                w.single("fleetComposition", "HOMOGENEOUS")
            with w.vehicles():
                for vehicle in self.vehicles:
                    vehicle.serialize(w)
            with w.vehicleTypes():
                for t in self.vehicle_types:
                    t.serialize(w)
            with w.services():
                for service in self.services:
                    service.serialize(w)
        with open(filename, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(w.source())


class ResultReader:
    @staticmethod
    def read_line(line):
        if set(line) != {"+", "-"}:
            raise TypeError(line)

    @staticmethod
    def read_title(line):
        return list(map(str.strip, line.split("|")))

    @staticmethod
    def read_capition(line):
        if not (line[0] == line[-1] == "|"):
            raise TypeError
        return line[1:-1].strip()

    @staticmethod
    def read_row(line):
        if not line.startswith("|"):
            return None
        row = map(str.strip, line.split("|"))
        row = [np.nan if col == "undef" else col for col in row]
        return row

    @staticmethod
    def read(f):
        data = {}
        current = {}
        status = 0
        num_vehicles = 0
        for line in f:
            line = line.strip("\n")
            if line.startswith("SLF4J"):
                continue
            if status % 2 == 0:
                ResultReader.read_line(line)
                status += 1
            elif status == 1:
                current['caption'] = ResultReader.read_capition(line)
                status += 1
            elif status == 3:
                current['title'] = ResultReader.read_title(line)
                current['rows'] = []
                status += 1
            elif status == 5:
                row = ResultReader.read_row(line)
                if row:
                    current['rows'].append(row)
                elif num_vehicles > 1:
                    num_vehicles -= 1
                else:
                    df = pd.DataFrame(current['rows'], columns=current['title'])
                    data[current['caption']] = df
                    if current['caption'] == "solution":
                        num_vehicles = int(df.iloc[1, 2])
                    current = {}
                    status = 0
        return data


if __name__ == '__main__':
    import os
    problem = Problem()
    problem.add_vehicle_type(VehicleType("Solomon", 200, {'fixed': 500, 'distance': 1.0, 'time': 0.0}))
    problem.add_vehile(Vehicle('0', "Solomon", (40.0, 50.0), (70.0, 10.0), (0, 1236.0)))
    problem.add_service(Service('35', 'pickup', 'location1', (5.0, 35.0), 10, 90.0, [(0, "1.79769E308")]))
    problem.to_xml("test.xml")
    result = os.popen("java -jar Jspirit-core-1.0-SNAPSHOT.jar test.xml")
    data = ResultReader.read(result)
    for key, value in data.items():
        print(key)
        print(value)
        print("------------------")



