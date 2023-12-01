package com.service.pcaccess;

import lombok.*;
import java.time.LocalDateTime;
import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "pc_access", schema = "insidoors")
public class PcAccess {
    
    private @Id @Column(name="id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="user_id")
    private int userId;
    
    @Column(name="access_date_time")
    private LocalDateTime accessDateTime;

    @Column(name="machine_name")
    private String machineName;

    @Column(name="machine_location")
    private String machineLocation;

    @Column(name="log_on_off")
    private String logOnOff;

    @Column(name="suspect")
    private int suspect;

    @Column(name="working_hours")
    private int workingHours;

    @Column(name="machine_lat")
    private double machineLat;

    @Column(name="machine_long")
    private double machineLong;
}
