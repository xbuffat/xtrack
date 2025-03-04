#ifndef XTRACK_MONITORS_H
#define XTRACK_MONITORS_H

/*gpufun*/
void ParticlesMonitor_track_local_particle(ParticlesMonitorData el, 
					   LocalParticle* part0){

    int64_t const start_at_turn = ParticlesMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = ParticlesMonitorData_get_stop_at_turn(el);
    int64_t const part_id_start = ParticlesMonitorData_get_part_id_start(el);
    int64_t const part_id_end= ParticlesMonitorData_get_part_id_end(el);
    int64_t const ebe_mode = ParticlesMonitorData_get_ebe_mode(el);
    ParticlesData data = ParticlesMonitorData_getp_data(el);

    int64_t n_turns_record = stop_at_turn - start_at_turn;

    //start_per_particle_block (part0->part)
	int64_t at_turn;
	if (ebe_mode){
		at_turn = LocalParticle_get_at_element(part);
	}
	else{
		at_turn = LocalParticle_get_at_turn(part);
	}
	if (at_turn>=start_at_turn && at_turn<stop_at_turn){
	    int64_t const particle_id = LocalParticle_get_particle_id(part);
	    if (particle_id<part_id_end && particle_id>=part_id_start){
	    	int64_t const store_at = 
		    n_turns_record * (particle_id - part_id_start)
		    + at_turn - start_at_turn;
	    	LocalParticle_to_Particles(part, data, store_at, 0);
	    }
	}
    //end_per_particle_block


}

#endif
