// Copyright 2024 Erin Lucassen (Delft University of Technology)
//                Robin Ferede (Delft University of Technology)
//                Stavrow Bahnam (Delft University of Technology)
//                Till Blaha (Delft University of Technology)
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/stat.h>

#include <fstream>
#include <sys/time.h>
#include <chrono>
#include <ctime>
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;


#include <iostream>
// to use the std:min function
#include <algorithm>
#include "pi-protocol.h"
#include "pi-messages.h"

#include <stdlib.h>
#include <arpa/inet.h>
#include <math.h>

#define OPTITRACK_PORT 5005
#define SETPOINT_PORT 5006
#define KEYBOARD_PORT 5007
#define MAX_BUFFER_SIZE 1024
#define PI_MSG_PAYLOAD_OFFSET (PI_MSG_ID_BYTES + PI_MSG_PAYLOAD_LEN_BYTES)

// hypersimple on-demand status updates similar to dd
// https://en.wikipedia.org/wiki/C_signal_handling
// idea: when sending SIGUSR1 -- print messages
//       when sending SIGUSR2 -- print statistics
// TODO: fork this program into two threads so that the main program isn't 
//       interrupted for stats/msgs. Neglect membar's, because we dont care if 
//       sometimes messages are garbled during the print and this hopefully 
//       wont occur too often
#include <signal.h>
#include <stdlib.h>

static float gyro[3];
static float accel[3];
static float omega[4];

static void catch_function(int signo) {
    switch(signo) {
        case SIGUSR1:
            piPrintMsgs(&printf);
            break;
        case SIGUSR2:
            piPrintStats(&printf);
            break;
    }
}

int openSerialPort(const char* port, int baudrate) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("openSerialPort: Unable to open port");
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);
    options.c_cflag = baudrate | CS8 | CLOCAL | CREAD;
    options.c_iflag = IGNPAR;
    options.c_oflag = 0;
    options.c_lflag = 0;
    tcflush(fd, TCIFLUSH);
    tcsetattr(fd, TCSANOW, &options);

    return fd;
}

int openUdpPort(const uint16_t port) {
    int sockfd;
    struct sockaddr_in server_addr;

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Make socket non-blocking
    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Bind socket
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    return sockfd;
}

int serialPortFd; // needs to be global to be used inside serialWriter

void serialWriter(uint8_t byte)
{
    //printf("%d", serialPortFd);
    write(serialPortFd, &byte, 1);
}

float ntohf(float in) {
    uint32_t dummy;
    dummy = ntohl(*(uint32_t*)&(in));
    return *(float*)(&dummy);
}

// for external pose
typedef struct pose_s {
    uint64_t timeUs;
    float x;
    float y;
    float z;
    float qx;
    float qy;
    float qz;
    float qw;
} pose_t;

typedef struct pose_der_s {
    uint64_t timeUs;
    float x;
    float y;
    float z;
    float wx;
    float wy;
    float wz;
} pose_der_t;

pose_t pose;
pose_der_t pose_der;

static pi_parse_states_t piParseStates = {0};

// functions for writing messages to csv file
FILE* createLogFile() {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[200];

    // Get the home directory
    const char* homeDir = getenv("HOME");
    if (homeDir == NULL) {
        fprintf(stderr, "Cannot get HOME directory\n");
        return NULL;
    }

    // Construct the full path for the log directory and file
    sprintf(filename, "%s/logs/imu-%d-%d-%d-%d-%d-%d.csv", homeDir, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string filename_str(filename);

    // Check if the directory exists, if not create it
    struct stat st = {0};
    std::string logDir = std::string(homeDir) + "/logs";
    if (stat(logDir.c_str(), &st) == -1) {
        if (mkdir(logDir.c_str(), 0700) != 0) {
            perror("mkdir failed");
            return NULL;
        }
    }

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("fopen failed");
    } else {
        printf("Logging data to %s\n", filename);
    }
    return file;
}

void writeCsvHeader(FILE* file) {
    // time
    fprintf(file, "time_us,");
    // piMsgImu
    fprintf(file, "imu_time_us, imu_roll, imu_pitch, imu_yaw, imu_x, imu_y, imu_z, omega_0, omega_1, omega_2, omega_3");
    // piMsgExternalPose
    fprintf(file, "ext_time_us, ext_x, ext_y, ext_z, ext_vx, ext_vy, ext_vz, ext_qw, ext_qx, ext_qy, ext_qz,");
#ifdef ORIN
    // piMsgVioPose
    fprintf(file, "vio_time_us, vio_x, vio_y, vio_z, vio_vx, vio_vy, vio_vz, vio_qw, vio_qx, vio_qy, vio_qz, vio_p, vio_q, vio_r\n");
#endif
}


void writeCsvRow(FILE* file) {
    // get current time us
    auto now = system_clock::now();
    auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    // static uint64_t time_us_start = now_us.time_since_epoch().count();
    // uint32_t time_us = (uint32_t)(now_us.time_since_epoch().count() - time_us_start);
    uint64_t time_us = now_us.time_since_epoch().count();
    
    // time
    fprintf(file, "%lu,", time_us);

    // piMsgImu
    if (piMsgEkfInputsRx != NULL) {
        //fprintf(file, "%u, %f, %f, %f, %f, %f, %f, ",
        //    piMsgImuRx->time_us, piMsgImuRx->roll, piMsgImuRx->pitch,
        //    piMsgImuRx->yaw, piMsgImuRx->x, piMsgImuRx->y, piMsgImuRx->z);
        fprintf(file, "%u, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f",
            piMsgEkfInputsRx->time_us,
            gyro[0], gyro[1], gyro[2],
            accel[0], accel[1], accel[2],
            omega[0], omega[1], omega[2], omega[3]
            );
    } else {
        fprintf(file, "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0");
    }
    // piMsgExternalPose
    // static uint64_t ext_time_us_start = pose.timeUs;
    // uint32_t ext_time_us = (uint32_t)(pose.timeUs - ext_time_us_start);
    uint64_t ext_time_us = pose.timeUs;
    fprintf(file, "%lu, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, ", ext_time_us, piMsgExternalPoseTx.ned_x, piMsgExternalPoseTx.ned_y, piMsgExternalPoseTx.ned_z, piMsgExternalPoseTx.ned_xd, piMsgExternalPoseTx.ned_yd, piMsgExternalPoseTx.ned_zd, piMsgExternalPoseTx.body_qi, piMsgExternalPoseTx.body_qx, piMsgExternalPoseTx.body_qy, piMsgExternalPoseTx.body_qz);
#ifdef ORIN
    fprintf(file, "%u, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f ", piMsgVioPoseTx.time_us, piMsgVioPoseTx.x, piMsgVioPoseTx.y, piMsgVioPoseTx.z, piMsgVioPoseTx.vx, piMsgVioPoseTx.vy, piMsgVioPoseTx.vz, piMsgVioPoseTx.qw, piMsgVioPoseTx.qx, piMsgVioPoseTx.qy, piMsgVioPoseTx.qz, piMsgVioPoseTx.p, piMsgVioPoseTx.q, piMsgVioPoseTx.r);
#endif
    fprintf(file, "\n");
}

#ifdef ORIN
// STUFF FOR ROS2
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <mutex>
#include <thread>
#include <chrono>


void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // std::lock_guard<std::mutex> lock(odom_mutex);

    // get time in ms
    static uint32_t start_s = msg->header.stamp.sec;
    // difference of first time and current time
    uint32_t diff_s = msg->header.stamp.sec - start_s;
    // diff to ms
    uint32_t time_us = diff_s * 1000000 + msg->header.stamp.nanosec / 1e3;

    // VIO_POSE MESSAGE
    piMsgVioPoseTx.time_us = time_us;
    piMsgVioPoseTx.x = msg->pose.pose.position.x;
    piMsgVioPoseTx.y = msg->pose.pose.position.y;
    piMsgVioPoseTx.z = msg->pose.pose.position.z;
    piMsgVioPoseTx.vx = msg->twist.twist.linear.x;
    piMsgVioPoseTx.vy = msg->twist.twist.linear.y;
    piMsgVioPoseTx.vz = msg->twist.twist.linear.z;
    piMsgVioPoseTx.qw = msg->pose.pose.orientation.w;
    piMsgVioPoseTx.qx = msg->pose.pose.orientation.x;
    piMsgVioPoseTx.qy = msg->pose.pose.orientation.y;
    piMsgVioPoseTx.qz = msg->pose.pose.orientation.z;
    piMsgVioPoseTx.p = msg->twist.twist.angular.x;
    piMsgVioPoseTx.q = msg->twist.twist.angular.y;
    piMsgVioPoseTx.r = msg->twist.twist.angular.z;

    // print content of piMsgVioPoseTx
    // printf("VIO_POSE: time_us: %u, x: %f, y: %f, z: %f, vx: %f, vy: %f, vz: %f, qw: %f, qx: %f, qy: %f, qz: %f\n", piMsgVioPoseTx.time_us, piMsgVioPoseTx.x, piMsgVioPoseTx.y, piMsgVioPoseTx.z, piMsgVioPoseTx.vx, piMsgVioPoseTx.vy, piMsgVioPoseTx.vz, piMsgVioPoseTx.qw, piMsgVioPoseTx.qx, piMsgVioPoseTx.qy, piMsgVioPoseTx.qz);

    piSendMsg(&piMsgVioPoseTx, &serialWriter);
    printf("received VIO_POSE \n");
}
#endif


int main(int argc, char** argv) {
    // enable logging if the first argument is -l
    bool logging = (argc == 2) && !strcmp(argv[1], "-l"); 

#ifdef ORIN
    const char* serialPort = "/dev/ttyTHS1";
#else
    const char* serialPort = "/dev/ttyAMA0";
#endif
    int baudrate = B921600;

    serialPortFd = openSerialPort(serialPort, baudrate);
    if (serialPortFd == -1) {
        return EXIT_FAILURE;
    }

    // register signal handlers (https://en.wikipedia.org/wiki/C_signal_handling)
    if (signal(SIGUSR1, catch_function) == SIG_ERR) {
        fputs("An error occurred while setting a signal handler.\n", stderr);
        return EXIT_FAILURE;
    }
    if (signal(SIGUSR2, catch_function) == SIG_ERR) {
        fputs("An error occurred while setting a signal handler.\n", stderr);
        return EXIT_FAILURE;
    }

    uint8_t piBuffer[PI_MAX_PACKET_LEN];

    // open udps
    int optitrackFd = openUdpPort(OPTITRACK_PORT);
    printf("Server listening on port %d for Optitrack...\n", OPTITRACK_PORT);
    static constexpr size_t OPTITRACK_BUFFER_SIZE = sizeof(unsigned int) + sizeof(pose_t) + sizeof(pose_der_t);
    uint8_t optitrackBuffer[OPTITRACK_BUFFER_SIZE];

    int setpointFd = openUdpPort(SETPOINT_PORT);
    printf("Server listening on port %d for Setpoints...\n", SETPOINT_PORT);
    uint8_t setpointBuffer[PI_MSG_POS_SETPOINT_PAYLOAD_LEN];

    int keyboardFd = openUdpPort(KEYBOARD_PORT);
    printf("Server listening on port %d for Keystrokes...\n", KEYBOARD_PORT);
    uint8_t keyboardBuffer[PI_MSG_KEYBOARD_PAYLOAD_LEN];

    // create imu log file and write to header
    FILE* imuLogFile = NULL;
    if (logging) {
        imuLogFile = createLogFile();
        writeCsvHeader(imuLogFile);
    }

#ifdef ORIN
    // ROS2 stuff for subscribing to odometry messages from ZED camera
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("zed_odometry_listener");

    auto subscription = node->create_subscription<nav_msgs::msg::Odometry>(
        "/zed/zed_node/odom", 10, odomCallback);

    // Start a separate thread for ROS2 spinning
    std::thread ros_thread([&node]() {
        rclcpp::spin(node);
    });
    // --------------------------------------------------------------------
#endif

    while (true) {
        // Q1: doesnt this add a lot of delay? Because the buffer is only filled once
        // PI_MAX_PACKET_LENGTH is read?

        // Q2: Is the UART TX buffer even big enough?
        // answer, seems like the PL011 buffer is 32byte deep. termios wraps it
        // in a PAGE_SIZE deep buffer (4096bytes), so this is fine at least
        ssize_t numBytes = read(serialPortFd, piBuffer, PI_MAX_PACKET_LEN);
        bool newMessage = false;
        if (numBytes) {
            for (int i=0; i < numBytes; i++) {
                if (piParse(&piParseStates, piBuffer[i]) == PI_MSG_EKF_INPUTS_ID) {
                    newMessage = true;

                    gyro[0]   = ((float)piMsgEkfInputsRx->p) * (2000.f * M_PI / 180.f) / (1 << 15);
                    gyro[1]   = ((float)piMsgEkfInputsRx->q) * (2000.f * M_PI / 180.f) / (1 << 15);
                    gyro[2]   = ((float)piMsgEkfInputsRx->r) * (2000.f * M_PI / 180.f) / (1 << 15);
                    accel[0]  = ((float)piMsgEkfInputsRx->x) * (9.81f / 2048);
                    accel[1]  = ((float)piMsgEkfInputsRx->y) * (9.81f / 2048);
                    accel[2]  = ((float)piMsgEkfInputsRx->z) * (9.81f / 2048);
                    omega[0] = ((float)piMsgEkfInputsRx->omega1);
                    omega[1] = ((float)piMsgEkfInputsRx->omega2);
                    omega[2] = ((float)piMsgEkfInputsRx->omega3);
                    omega[3] = ((float)piMsgEkfInputsRx->omega4);

                    if (logging) {
                        writeCsvRow(imuLogFile);
                    }
                }
            }
        }

        if ((piMsgEkfInputsRxState == PI_MSG_RX_STATE_NONE) || (!newMessage)) {
            // cannot go on, no time information to timestamp gps msgs, or setpoints
            //usleep(250); // reduce CPU load a bit
            continue;
        } else {
            newMessage = false;
        }


        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        int optitrackBytes = recvfrom(optitrackFd, (uint8_t *)(optitrackBuffer), OPTITRACK_BUFFER_SIZE, 0, (struct sockaddr *)&client_addr, &client_addr_len);

        if (optitrackBytes > 0) {
            // we have optitrack copy from the buffer into the structs
            memcpy((uint8_t *)(&pose), optitrackBuffer+sizeof(unsigned int), sizeof(pose_t));
            memcpy((uint8_t *)(&pose_der), optitrackBuffer+sizeof(unsigned int)+sizeof(pose_t), sizeof(pose_der_t));

            // proceed to send Fake GPS and External Pose
            piMsgFakeGpsTx.time_us = piMsgEkfInputsRx->time_us;
            static constexpr double CYBERZOO_LAT = 51.99071002805145;
            static constexpr double CYBERZOO_LON = 4.376727452462819;
            static constexpr double RE = 6378137.;
            piMsgFakeGpsTx.lat = (int) 1e7 * 
                (CYBERZOO_LAT + 180. / M_PI * (pose.x / RE));
            piMsgFakeGpsTx.lon = (int) 1e7 *
                (CYBERZOO_LON + 180 / M_PI * (pose.y / RE) / cos(CYBERZOO_LON * M_PI / 180.));
            piMsgFakeGpsTx.altCm = (int) (pose.z * 100.f);
            piMsgFakeGpsTx.hdop =  (short) 150;
            piMsgFakeGpsTx.groundSpeed = (short) (hypotf(pose_der.x, pose_der.y) * 100.f);
            piMsgFakeGpsTx.groundCourse = (short) (1800.f * atan2(pose_der.x, pose_der.y) / M_PI);
            piMsgFakeGpsTx.numSat = 8;
            piSendMsg(&piMsgFakeGpsTx, &serialWriter);

            piMsgExternalPoseTx.time_us = piMsgEkfInputsRx->time_us;
            piMsgExternalPoseTx.ned_x   = pose.x;
            piMsgExternalPoseTx.ned_y   = pose.y;
            piMsgExternalPoseTx.ned_z   = pose.z;
            piMsgExternalPoseTx.ned_xd  = pose_der.x;
            piMsgExternalPoseTx.ned_yd  = pose_der.y;
            piMsgExternalPoseTx.ned_zd  = pose_der.z;
            piMsgExternalPoseTx.body_qi = pose.qw;
            piMsgExternalPoseTx.body_qx = pose.qx;
            piMsgExternalPoseTx.body_qy = pose.qy;
            piMsgExternalPoseTx.body_qz = pose.qz;
            piSendMsg(&piMsgExternalPoseTx, &serialWriter);
            printf("forwarded EXTERNAL_POSE \n");

            piMsgOffboardPoseTx.time_us = piMsgEkfInputsRx->time_us;
            piMsgOffboardPoseTx.x = 0.;
            piMsgOffboardPoseTx.y = 0.;
            piMsgOffboardPoseTx.z = 0.;
            piMsgOffboardPoseTx.qw = 1.;
            piMsgOffboardPoseTx.qx = 0.;
            piMsgOffboardPoseTx.qy = 0.;
            piMsgOffboardPoseTx.qz = 0.;
            piSendMsg(&piMsgOffboardPoseTx, &serialWriter);
            printf("forwarded OFFBOARD_POSE \n");
        }

        // ---- setpoints ----
        pi_POS_SETPOINT_t msgPosSetpoint;
        int setpointBytes = recvfrom(setpointFd, (uint8_t *)(setpointBuffer), PI_MSG_POS_SETPOINT_PAYLOAD_LEN, 0, (struct sockaddr *)&client_addr, &client_addr_len);

        if (setpointBytes > 0) {
            memcpy((uint8_t *)(&msgPosSetpoint)+PI_MSG_PAYLOAD_OFFSET, setpointBuffer, PI_MSG_POS_SETPOINT_PAYLOAD_LEN);
            piMsgPosSetpointTx.time_us = piMsgEkfInputsRx->time_us;

            piMsgPosSetpointTx.ned_x  = ntohf(msgPosSetpoint.ned_x);
            piMsgPosSetpointTx.ned_y  = ntohf(msgPosSetpoint.ned_y);
            piMsgPosSetpointTx.ned_z  = ntohf(msgPosSetpoint.ned_z);
            piMsgPosSetpointTx.ned_xd = ntohf(msgPosSetpoint.ned_xd);
            piMsgPosSetpointTx.ned_yd = ntohf(msgPosSetpoint.ned_yd);
            piMsgPosSetpointTx.ned_zd = ntohf(msgPosSetpoint.ned_zd);
            piMsgPosSetpointTx.yaw = ntohf(msgPosSetpoint.yaw);
            piSendMsg(&piMsgPosSetpointTx, &serialWriter);
            printf("received SETPOINT \n");
        }

        // ---- keyboard ----
        pi_KEYBOARD_t msgKeyboard;
        int keyboardBytes = recvfrom(keyboardFd, (uint8_t *)(keyboardBuffer), PI_MSG_KEYBOARD_PAYLOAD_LEN, 0, (struct sockaddr *)&client_addr, &client_addr_len);

        if (keyboardBytes > 0) {
            memcpy((uint8_t *)(&msgKeyboard)+PI_MSG_PAYLOAD_OFFSET, keyboardBuffer, PI_MSG_KEYBOARD_PAYLOAD_LEN);
            piMsgKeyboardTx.time_us = piMsgEkfInputsRx->time_us;
            piMsgKeyboardTx.key = msgKeyboard.key;
            piSendMsg(&piMsgKeyboardTx, &serialWriter);
            printf("received KEYBOARD \n");

        }
    }

    close(serialPortFd);
    close(optitrackFd);
    close(setpointFd);
    close(keyboardFd);

    return 0;
}