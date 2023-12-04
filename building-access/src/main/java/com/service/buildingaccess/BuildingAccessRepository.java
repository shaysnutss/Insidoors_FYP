package com.service.buildingaccess;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

@Repository
public interface BuildingAccessRepository extends JpaRepository<BuildingAccess, Long> {
    
    @Query(value = "SELECT * FROM building_access WHERE ID > ?1", nativeQuery = true)
    List<BuildingAccess> findAllAfterId(Long desiredId);
}
