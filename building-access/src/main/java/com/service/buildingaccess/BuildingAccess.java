package com.service.buildingaccess;

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
@Table(name = "building_access", schema = "insidoors")
public class BuildingAccess {
    
    private @Id @Column(name="id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="user_id")
    private int userId;
    
    @Column(name="access_date_time")
    private LocalDateTime accessDateTime;

    @Column(name="direction")
    private String direction;

    @Column(name="status")
    private String status;

    @Column(name="office_location")
    private String officeLocation;

    @Column(name="suspect")
    private int suspect;

    @Column(name="office_lat")
    private double officeLat;

    @Column(name="office_long")
    private double officeLong;

    @Column(name="attempts")
    private int attempts;
}
