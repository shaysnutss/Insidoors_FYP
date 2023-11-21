package com.service.proxylog;

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
@Table(name = "proxy_log", schema = "insidoors")
public class ProxyLog {

    private @Id @Column(name="id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="user_id")
    private int userId;
    
    @Column(name="access_date_time")
    private LocalDateTime accessDateTime;

    @Column(name="machine_name")
    private String machineName;

    @Column(name="url")
    private String url;

    @Column(name="category")
    private String category;
    
    @Column(name="bytes_in")
    private int bytesIn;

    @Column(name="bytes_out")
    private int bytesOut;

    @Column(name="suspect")
    private int suspect;
}
