package com.service.behavioralanalysisservice;

import lombok.*;
import jakarta.persistence.*;

@Entity
@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@EqualsAndHashCode
@Table(name = "behavioral_analysis_service", schema = "behavioral_analysis")
public class BehavioralAnalysisService {
    
    private @Id @Column(name="ba_id") @GeneratedValue (strategy = GenerationType.IDENTITY) Long id;

    @Column(name="employee_id")
    private int employeeId;
    
    @Column(name="risk_rating")
    private int riskRating;

    @Column(name="suspected_cases")
    private int suspectedCases;

    public BehavioralAnalysisService(int employeeId, int riskRating, int suspectedCases) {
        this.employeeId = employeeId;
        this.riskRating = riskRating;
        this.suspectedCases = suspectedCases;
    }

}
